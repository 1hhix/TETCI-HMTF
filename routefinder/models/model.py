from typing import Any, Optional

import torch
import torch.nn as nn
from collections import deque
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
import copy
from routefinder.models.reward_normalization import (
    CumulativeMean,
    ExponentialMean,
    NoNormalization,
    ZNormalization,
)
import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
log = get_pylogger(__name__)


class RouteFinderBase(POMO):
    """
    Main RouteFinder RL model based on POMO.
    This automatically include the Mixed Batch Training (MBT) from the environment.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        automatic_optimization: bool=True,
        if_norm: bool=True,
        extra_loss: bool=False,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)
        self.if_norm=if_norm
        alpha = kwargs.pop("alpha", 0.1)
        epsilon = kwargs.pop("epsilon", 1e-5)
        normalize_reward = kwargs.pop("normalize_reward", "none")
        self.norm_operation = kwargs.pop("norm_operation", "div")  # "div" or "sub"

        # Initialize with the shared baseline
        super(RouteFinderBase, self).__init__(env, policy, **kwargs)
        self.automatic_optimization=automatic_optimization
        allowed_normalizations = [
            "cumulative",
            "exponential",
            "none",
            "normal",
            "z",
            "z-score",
        ]
        assert (
            normalize_reward in allowed_normalizations
        ), f"normalize_reward must lie in {allowed_normalizations}."

        if normalize_reward == "cumulative":
            self.normalization = CumulativeMean()
        elif normalize_reward == "exponential":
            self.normalization = ExponentialMean(alpha=alpha)
        elif normalize_reward == "none":
            self.normalization = NoNormalization()
        elif normalize_reward in ["normal", "z", "z-score"]:
            self.normalization = ZNormalization(alpha=alpha, epsilon=epsilon)
        else:
            raise NotImplementedError("Normalization not implemented")
        self.extra_loss=extra_loss
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        costs_bks = batch.get("costs_bks", None)

        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            normalized_reward, norm_vals,cost_step = self.normalization(
                td=unbatchify(x=td, shape=n_aug),
                rewards=reward,
                operation=self.norm_operation,
                if_norm=self.if_norm
            )
            out.update({"norm_vals": norm_vals, "norm_reward": normalized_reward})
            self.calculate_loss(td, batch, out, normalized_reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            max_norm_reward, _ = normalized_reward.max(dim=-1)
            out.update({"max_reward": max_reward, "max_norm_reward": max_norm_reward})
            if cost_step is not None:
                out.update(cost_step)

            if (not self.automatic_optimization) and (not self.extra_loss):
                opt=self.optimizers()
                opt.zero_grad()
                self.manual_backward(out['loss'])
                self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                opt.step()
                if self.trainer.is_last_batch:
                    self.lr_schedulers().step()

        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {
                            "best_multistart_actions": gather_by_index(
                                actions, max_idxs, dim=max_idxs.dim()
                            )
                        }
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                # If costs_bks is available, we calculate the gap to BKS
                if costs_bks is not None:
                    # Rewards are negative costs; normalize against the absolute BKS cost.
                    gap_to_bks = (
                        100
                        * (-max_aug_reward - torch.abs(costs_bks))
                        / torch.abs(costs_bks)
                    )
                    out.update({"gap_to_bks": gap_to_bks})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

            if out.get("gap_to_bks", None) is None:
                out.update({"gap_to_bks": 100})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        advantage = self.advantage_scaler(advantage)
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out


class RouteFinderMoE(RouteFinderBase):
    """RouteFinder with MoE model as the policy as in MVMoE (https://github.com/RoyalSkye/Routing-MVMoE).
    This includes the Mixed Batch Training (MBT) from the environment.
    Note that additional losses are added to the loss function for MoE during training.
    Note that to use the new embeddings, you should pass them to the new policy via:
    - init_embedding: MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
    - context_embedding: MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

    Ref for MVMoE:
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        **kwargs,
    ):
        from routefinder.models.baselines.mvmoe.policy import (
            MVMoELightPolicy,
            MVMoEPolicy,
        )

        assert isinstance(
            policy, (MVMoEPolicy, MVMoELightPolicy)
        ), "policy must be an instance of MVMoEPolicy or MVMoELightPolicy"

        super(RouteFinderMoE, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # Shared step from POMO
        out = super(RouteFinderMoE, self).shared_step(
            batch, batch_idx, phase, dataloader_idx
        )

        loss = out.get("loss", None)

        if loss is not None:
            if hasattr(self.policy.encoder.init_embedding, "moe_loss"):
                moe_loss_init_embeds = self.policy.encoder.init_embedding.moe_loss
            else:
                moe_loss_init_embeds = 0

            moe_loss_layers = 0
            for layer in self.policy.encoder.net.layers:
                if hasattr(layer, "moe_loss"):
                    moe_loss_layers += layer.moe_loss
                else:
                    moe_loss_layers += 0

            if hasattr(self.policy.decoder.pointer, "moe_loss"):
                moe_loss_decoder = self.policy.decoder.pointer.moe_loss
            else:
                moe_loss_decoder = 0

            # Sum losses and save in out for backpropagation
            moe_loss = moe_loss_init_embeds + moe_loss_layers + moe_loss_decoder
            out["loss"] = loss + moe_loss

        return out


class RouteFinderSingleVariantSampling(RouteFinderBase):
    """This is the default sampling method for MVMoE and MTPOMO
    (without Mixed-Batch Training) as first proposed in MTPOMO (https://arxiv.org/abs/2402.16891)

    The environment generates by default all the features,
    and we subsample them at each batch to train the model (i.e. we select a subset of the features).

    For example: we always sample OVRPBLTW (with all features) and we simply take a subset of them at each batch.

    Note we removed the support for single_feat_otw (original MVMoE more restricted setting) since it is not used
    in the experiments in Foundation Model settings, however it can be added back if needed
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        preset: str = "all",
        **kwargs,
    ):
        assert (
            not env.generator.subsample
        ), "The env generator must not subsample the features, this is done in the `shared_step` method"

        super(RouteFinderSingleVariantSampling, self).__init__(
            env,
            policy,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = batch

        # variant subsampling: given a batch with *all* features, we subsample a part of them
        if phase == "train":

            # Sample single variant (i.e which features to *remove* with 50% probability)
            indices = torch.bernoulli(torch.tensor([0.5] * 4))

            # Process the indices
            if indices[0] == 1:  # Remove open
                td["open_route"] &= False
            if indices[1] == 1:  # Remove time window
                td["time_windows"][..., 0] *= 0
                td["time_windows"][..., 1] += float("inf")
                td["service_time"] *= 0
            if indices[2] == 1:  # Remove distance limit
                td["distance_limit"] += float("inf")
            if indices[3] == 1:  # Remove backhaul
                td.set("demand_linehaul", td["demand_linehaul"] + td["demand_backhaul"])
                td.set("demand_backhaul", torch.zeros_like(td["demand_backhaul"]))

        return super(RouteFinderSingleVariantSampling, self).shared_step(
            td, batch_idx, phase, dataloader_idx
        )

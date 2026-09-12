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
from hmtf.models.reward_normalization import (
    CumulativeMean,
    ExponentialMean,
    NoNormalization,
    ZNormalization,
)
import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize
log = get_pylogger(__name__)

class HMTF(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        if_norm:bool=True,
        use_mtl: bool=False,
        buffer_len: int=300,
        c: float=0.4,
        **kwargs,
    ):
        self.if_norm = if_norm
        self.use_mtl = use_mtl
        self.n_tasks=2
        self.c = c
        self.save_hyperparameters(logger=False)

        alpha = kwargs.pop("alpha", 0.1)
        epsilon = kwargs.pop("epsilon", 1e-5)
        normalize_reward = kwargs.pop("normalize_reward", "none")
        self.norm_operation = kwargs.pop("norm_operation", "div")

        # Initialize with the shared baseline
        super(HMTF, self).__init__(env, policy, **kwargs)

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
        self.buffer_len=buffer_len
        self.grad_buffer=deque(maxlen=self.buffer_len)
        self.loss_buffer=deque(maxlen=self.buffer_len)
        self.pre_loss=None


    def get_G_wrt_shared(self,losses, shared_params,opt, update_decoder_grads=False):
        grads = []
        for i,cur_loss in enumerate(losses):
            if not update_decoder_grads:
                opt.zero_grad()
                grad = torch.cat([p.flatten() if p is not None else torch.zeros_like(shared_params[i]).flatten()
                                  for i, p in enumerate(torch.autograd.grad(cur_loss, shared_params,
                                                               retain_graph=True, allow_unused=True))])
            else:
                opt.zero_grad()
                self.manual_backward(cur_loss,retain_graph=True)
                grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                                  for p in shared_params])

            grads.append(grad)
        return grads
    @staticmethod
    def set_shared_grad(parameters, grad_vec):
        offset = 0
        for p in list(parameters):
            _offset = offset + p.shape.numel()
            p.grad = grad_vec[offset:_offset].view_as(p).clone().detach()
            offset = _offset

    def gb(self, grads,shared_parameters, alpha=0.5, rescale=1):
        eps=1e-24
        x_start = np.ones(self.n_tasks) / self.n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
        b = x_start.copy()
        GG = (grads).t().mm(grads).cpu()  # [num_tasks, num_tasks]
        A = GG.numpy()
        g0_norm = (GG.mean() + eps).sqrt()
        g_1,g_2=grads[:,0],grads[:,1]
        if torch.dot(g_1,g_2)>0:
            GG = (grads).t().mm(grads)  # [num_tasks, num_tasks]
            A = GG.cpu().numpy()
            g0_norm = (GG.mean().cpu() + eps).sqrt()

        else:
            grads=grads/(grads+ eps).norm(dim=0)
            return (1+alpha)*grads.mean(dim=1)/ (1 + alpha ** 2),[0.5,0.5]

        c = (alpha * g0_norm + eps).item()
        def objfn(x):
            return (
                x.reshape(1, self.n_tasks).dot(A).dot(b.reshape(self.n_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, self.n_tasks).dot(A).dot(x.reshape(self.n_tasks, 1))
                    + eps
                )
            ).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        self.ww_pre=ww.clone()
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + eps)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g,ww
        elif rescale == 1:
            return g / (1 + alpha ** 2),ww
        else:
            return g / (1 + alpha),ww


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
            out.update(cost_step)
            self.calculate_loss(td, batch, out, normalized_reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            max_norm_reward, _ = normalized_reward.max(dim=-1)
            out.update({"max_reward": max_reward, "max_norm_reward": max_norm_reward})
            if not self.automatic_optimization:
                opt=self.optimizers()
                if not self.use_mtl:
                    opt.zero_grad()
                    self.manual_backward(out['loss'])
                    self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                    opt.step()
                else:
                    parameters=list(self.policy.encoder.parameters())
                    opt.zero_grad()
                    self.manual_backward(out['loss'])
                    grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                                for p in parameters])
                    use_his=torch.dot(grad,self.grad_buffer[-1])<=0 if len(self.grad_buffer)>=1 else False
                    if len(self.grad_buffer)<self.buffer_len:
                        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                        opt.step()
                        g=grad.clone()
                    else:
                        if use_his:
                            curren_loss=torch.stack([loss_p for loss_p in self.loss_buffer]).mean()
                            his_grad=torch.stack([g for g in self.grad_buffer],dim=1).mean(dim=-1,keepdim=True)
                            grads=torch.cat((grad.unsqueeze(1), self.c*(grad.norm()/his_grad.norm())*his_grad),dim=-1)
                            g,ww = self.gb(grads,parameters,alpha=self.c, rescale=1)
                            self.set_shared_grad(parameters,g)
                            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                            opt.step()
                            self.pre_loss=curren_loss.clone()
                            self.grad_buffer = deque(list(self.grad_buffer)[int(self.buffer_len*0.2):], maxlen=self.buffer_len)
                            self.loss_buffer = deque(list(self.loss_buffer)[int(self.buffer_len*0.2):], maxlen=self.buffer_len)

                        else:
                            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                            opt.step()
                            g=grad.clone()

                    self.grad_buffer.append(g.clone().detach())
                    self.loss_buffer.append(out['loss'].clone().detach())



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
        if self.automatic_optimization or not self.use_mtl:
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
        else:

            reinforce_loss = -(advantage * log_likelihood).mean()

            policy_out.update(
                {   "loss": reinforce_loss,
                    "reinforce_loss": 0,
                    "bl_loss": bl_loss,
                    "bl_val": bl_val,
                }
            )
        return policy_out

from typing import Any

from rl4co.envs.common.base import RL4COEnvBase

from routefinder.models.model import RouteFinderSingleVariantSampling

from .policy import MVMoELightPolicy, MVMoEPolicy


class MVMoE(RouteFinderSingleVariantSampling):
    """Original MVMoE model with single variant sampling at each batch"""

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: MVMoEPolicy,
        automatic_optimization: bool=True,
        **kwargs,
    ):
        assert isinstance(
            policy, (MVMoEPolicy, MVMoELightPolicy)
        ), "policy must be an instance of MVMoEPolicy or MVMoELightPolicy"

        super(MVMoE, self).__init__(
            env,
            policy,
            **kwargs,
        )
        self.automatic_optimization=automatic_optimization
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        out = super(MVMoE, self).shared_step(batch, batch_idx, phase, dataloader_idx)

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

            if hasattr(self.policy.decoder.pointer, "moe_loss"):
                moe_loss_decoder = self.policy.decoder.pointer.moe_loss
            else:
                moe_loss_decoder = 0

            moe_loss = moe_loss_init_embeds + moe_loss_layers + moe_loss_decoder
            out["loss"] = loss + moe_loss

            if not self.automatic_optimization:
                opt=self.optimizers()
                opt.zero_grad()
                self.manual_backward(out['loss'])
                self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                opt.step()
                if self.trainer.is_last_batch:
                    self.lr_schedulers().step()
        return out

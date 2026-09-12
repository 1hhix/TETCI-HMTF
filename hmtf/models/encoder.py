from typing import Tuple, Union

import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from torch import Tensor
from hmtf.models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder
from hmtf.models.nn.transformer import Normalization, TransformerBlock
from hmtf.models.nn.hyper_net import Hyper_Block_T
log = get_pylogger(__name__)


class DFM_encoder(nn.Module):
    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        use_hyper_encoder: bool = False,
        **transformer_kwargs,
    ):
        super(DFM_encoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding
        self.use_hyper_encoder=use_hyper_encoder
        encoder_block=Hyper_Block_T if self.use_hyper_encoder else TransformerBlock
        self.layers = nn.Sequential(
            *(
                encoder_block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        h=(init_h,td['task_embedding'].to(init_h.device)) if self.use_hyper_encoder else init_h

        for layer in self.layers:
            h = layer(h, mask)

        h = h[0] if self.use_hyper_encoder else h

        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        return h, init_h

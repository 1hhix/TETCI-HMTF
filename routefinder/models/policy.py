from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from routefinder.models.encoder import RouteFinderEncoder

from hmtf.models.encoder import DFM_encoder
from hmtf.models.decoder import FM_decoder
from hmtf.models.env_embeddings.mtvrp import (
    MTVRPContextEmbeddingRouteFinder,
    DFMContextEmbedding,
    DFM_InitEmbedding,
    MTVRPInitEmbeddingRouteFinder
)

log = get_pylogger(__name__)


class RouteFinderPolicy(AttentionModelPolicy):
    """
    Main RouteFinder policy based on the Transformer Architecture.
    We use the base AttentionModelPolicy for decoding (i.e. masked attention + pointer network)
    and our new RouteFinderEncoder for the encoder.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        parallel_gated_kwargs: dict = None,
        encoder_use_post_layers_norm: bool = False,
        encoder_use_prenorm: bool = False,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: MTVRPInitEmbeddingRouteFinder = None,
        context_embedding: MTVRPContextEmbeddingRouteFinder = None,
        extra_encoder_kwargs: dict = {},
        **kwargs,
    ):

        encoder = RouteFinderEncoder(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        super(RouteFinderPolicy, self).__init__(
            encoder=encoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )


class DFMPolicy_Hyper(AttentionModelPolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        parallel_gated_kwargs: dict = None,
        encoder_use_post_layers_norm: bool = False,
        encoder_use_prenorm: bool = False,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: DFM_InitEmbedding = None,
        context_embedding: DFMContextEmbedding = None,
        extra_encoder_kwargs: dict = {},
        use_hyper_encoder: bool = False,
        use_hyper_decoder: bool = False,
        decoder_time_step: bool = False,
        **kwargs,
    ):

        encoder = DFM_encoder(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            use_hyper_encoder=use_hyper_encoder,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)

        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        decoder = FM_decoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=None,
                mask_inner=True,
                out_bias_pointer_attn=False,
                linear_bias=False,
                use_graph_context=use_graph_context,
                check_nan=True,
                moe_kwargs=None,
                use_hyper_decoder=use_hyper_decoder,
                decoder_time_step=decoder_time_step,
            )

        super(DFMPolicy_Hyper, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )


USE_CUSTOMIZED_IMPL = False

if USE_CUSTOMIZED_IMPL:
    from .attention import Attention
    from .transformers_2d import Transformer2DModel
    from .hunyuan_transformer_2d import HunyuanDiT2DModel
else:
    from diffusers.models.attention import Attention
    from diffusers.models.transformers.transformer_2d import Transformer2DModel
    from diffusers.models.transformers.hunyuan_transformer_2d import HunyuanDiT2DModel

__all__ = ["Attention", "Transformer2DModel"]

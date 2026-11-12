"""sspc: online learned lossless compression with an ordered-mask MLP."""

from .codec import CodecConfig, compress, decompress
from .model import ModelConfig, OrderedMaskMLP

__all__ = ["CodecConfig", "ModelConfig", "OrderedMaskMLP", "compress", "decompress"]

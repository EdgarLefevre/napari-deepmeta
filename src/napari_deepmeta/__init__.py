try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._widget import DeepmetaDemoWidget, DeepmetaWidget

__all__ = ("DeepmetaWidget", "DeepmetaDemoWidget")

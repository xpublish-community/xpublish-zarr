"""Zarr-compatible REST API plugin for Xpublish."""

from xpublish_zarr.plugin import ZarrPlugin

try:
    from xpublish_zarr._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["ZarrPlugin", "__version__"]

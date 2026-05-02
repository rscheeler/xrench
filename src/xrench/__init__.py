"""xrench: to support RF and microwave design Python tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xrench")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0-dev"

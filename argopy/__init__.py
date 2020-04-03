try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("argopy").version
except Exception:
    # Local copy, not installed with setuptools, or setuptools is not available.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# Import facades:
from .fetchers import ArgoDataFetcher as DataFetcher
from .xarray import ArgoAccessor
from . import tutorial

# Other Import
from . import utilities
from .options import set_options

#
__all__ = (
    # Classes:
    "DataFetcher",
    "ArgoAccessor",
    # Top-level functions:
    "set_options",
    # Sub-packages,
    "utilities",
    "errors",
    # Constants
    "__version__"
)
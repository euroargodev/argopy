try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("argopy").version
except Exception:
    # Local copy, not installed with setuptools, or setuptools is not available.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# Import facades:
from .fetchers import ArgoDataFetcher as DataFetcher
from .fetchers import ArgoIndexFetcher as IndexFetcher

from .xarray import ArgoAccessor
from . import tutorial

# Other Import
from . import utilities
from .utilities import show_versions
from .options import set_options

#
__all__ = (
    # Classes:
    "DataFetcher",
    "IndexFetcher",
    "ArgoAccessor",
    # Top-level functions:
    "set_options",
    "show_versions",
    # Sub-packages,
    "utilities",
    "errors",
    "plotters",
    # Constants
    "__version__"
)
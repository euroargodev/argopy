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
from . import stores
from .utilities import show_versions, clear_cache
from .options import set_options
from .plotters import open_dashboard as dashboard

#
__all__ = (
    # Classes:
    "DataFetcher",
    "IndexFetcher",
    "ArgoAccessor",
    # Top-level functions:
    "set_options",
    "show_versions",
    "dashboard",
    "clear_cache",
    # Sub-packages,
    "utilities",
    "errors",
    "plotters",
    "stores",
    "tutorial",
    # Constants
    "__version__"
)

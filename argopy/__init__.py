try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("argopy").version
except Exception:
    # Local copy, not installed with setuptools, or setuptools is not available.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

# Loggers
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Import facades:
from .fetchers import ArgoDataFetcher as DataFetcher  # noqa: 
from .fetchers import ArgoIndexFetcher as IndexFetcher  # noqa: F402

from .xarray import ArgoAccessor  # noqa: F402
from . import tutorial  # noqa: F402

# Other Import
from . import utilities  # noqa: F402
from . import stores  # noqa: F402
from .utilities import show_versions, show_options, clear_cache  # noqa: F402
from .utilities import monitor_status as status  # noqa: F402
from .options import set_options  # noqa: F402
from .plotters import open_dashboard as dashboard  # noqa: F402

#
__all__ = (
    # Classes:
    "DataFetcher",
    "IndexFetcher",
    "ArgoAccessor",
    # Top-level functions:
    "set_options",
    "show_versions",
    "show_options",
    "dashboard",
    "status",
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

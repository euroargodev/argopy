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
# log.info("Welcome ! this is a INFO msg")
# log.debug("Welcome ! this is a DEBUG msg")
# log.warning("Welcome ! this is a WARNING msg")

# Import facades:
from .fetchers import ArgoDataFetcher as DataFetcher  # noqa: F401 isort:skip
from .fetchers import ArgoIndexFetcher as IndexFetcher  # noqa: F401 isort:skip

from .xarray import ArgoAccessor  # noqa: F401 isort:skip
from . import tutorial  # noqa: F401 isort:skip

# Other Import
from . import utilities  # noqa: F401 isort:skip
from . import stores  # noqa: F401 isort:skip
from .utilities import show_versions, clear_cache  # noqa: F401 isort:skip
from .utilities import monitor_status as status  # noqa: F401 isort:skip
from .options import set_options  # noqa: F401 isort:skip
from .plotters import open_dashboard as dashboard  # noqa: F401 isort:skip

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

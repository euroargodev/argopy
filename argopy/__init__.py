"""
Argopy library
"""

try:
    from importlib.metadata import version as _version
except ImportError:
    # if the fallback library is missing, we are doomed.
    from importlib_metadata import version as _version

try:
    __version__ = _version("argopy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"


# Loggers
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Import facades:
from .fetchers import ArgoDataFetcher as DataFetcher  # noqa: E402
from .fetchers import ArgoIndexFetcher as IndexFetcher  # noqa: E402

from .xarray import ArgoAccessor  # noqa: E402
from . import tutorial  # noqa: E402

# Other Import
from . import utilities  # noqa: E402
from . import stores  # noqa: E402
from . import errors  # noqa: E402
from . import plot  # noqa: E402
from .plot import dashboard, ArgoColors  # noqa: E402
from .utilities import show_versions, show_options, clear_cache, lscache  # noqa: E402
from .utilities import TopoFetcher, ArgoNVSReferenceTables, OceanOPSDeployments, ArgoDocs  # noqa: E402
from .utilities import monitor_status as status  # noqa: E402
from .options import set_options, reset_options  # noqa: E402
from .data_fetchers import CTDRefDataFetcher  # noqa: E402
from .stores import ArgoIndex  # noqa: E402
from .utils import compute  # noqa: E402, F401

#
__all__ = (
    # Top-level classes:
    "DataFetcher",
    "IndexFetcher",
    "ArgoAccessor",

    # Utilities promoted to top-level functions:
    "set_options",
    "reset_options",
    "show_versions",
    "show_options",
    "dashboard",
    "status",
    "clear_cache",
    "lscache",

    # Meta-data and other related dataset helpers class:
    "ArgoNVSReferenceTables",  # Class
    "OceanOPSDeployments",  # Class
    "CTDRefDataFetcher",  # Class
    "ArgoIndex",  # Class
    "ArgoDocs",  # Class
    "TopoFetcher",  # Class

    # Submodules:
    "utilities",
    "errors",
    "plot",
    "ArgoColors",  # Class
    # "plotters",  # Deprec, to be removed after 0.1.13
    "stores",
    "tutorial",
    # Constants
    "__version__"
)

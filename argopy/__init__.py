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

# Other Import
# from . import utils  # noqa: E402
from . import stores  # noqa: E402
from . import errors  # noqa: E402
from . import plot  # noqa: E402
from . import tutorial  # noqa: E402
from .plot import dashboard, ArgoColors  # noqa: E402
from .options import set_options, reset_options  # noqa: E402
from .data_fetchers import CTDRefDataFetcher  # noqa: E402
from .stores import ArgoIndex, ArgoFloat, gdacfs  # noqa: E402
from .utils import show_versions, show_options  # noqa: E402
from .utils import clear_cache, lscache  # noqa: E402
from .utils import MonitoredThreadPoolExecutor  # noqa: E402, F401
from .utils import monitor_status as status  # noqa: E402
from .related import TopoFetcher, OceanOPSDeployments, ArgoNVSReferenceTables, ArgoDocs, ArgoDOI  # noqa: E402
from .extensions import CanyonMED  # noqa: E402


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
    "ArgoDocs",  # Class
    "TopoFetcher",  # Class
    "ArgoDOI",  # Class

    # Advanced Argo data stores:
    "ArgoFloat",  # Class
    "ArgoIndex",  # Class
    "gdacfs",  # Class

    # Submodules:
    # "utils",
    "errors",
    "plot",
    "ArgoColors",  # Class
    "stores",
    "tutorial",

    # Argo xarray accessor extensions
    "CanyonMED",

    # Constants
    "__version__"
)

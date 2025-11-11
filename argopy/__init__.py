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
from argopy.fetchers import ArgoDataFetcher as DataFetcher  # noqa: E402
from argopy.fetchers import ArgoIndexFetcher as IndexFetcher  # noqa: E402

from argopy.xarray import ArgoAccessor  # noqa: E402

# Other Import
# from . import utils  # noqa: E402
# from argopy import errors  # noqa: E402
# from argopy import plot  # noqa: E402
import argopy.tutorial  # noqa: E402
from argopy.plot.dashboards import open_dashboard as dashboard
from argopy.plot.argo_colors import ArgoColors  # noqa: E402
from argopy.options import set_options, reset_options  # noqa: E402
from argopy.data_fetchers.erddap_refdata import Fetch_box as CTDRefDataFetcher  # noqa: E402
from argopy.stores.index.argo_index import ArgoIndex  # noqa: E402
from argopy.stores.float.argo_float import ArgoFloat  # noqa: E402
from argopy.stores.implementations.gdac import gdacfs  # noqa: E402
from argopy.utils.locals import show_versions, show_options  # noqa: E402
from argopy.utils.caching import clear_cache, lscache  # noqa: E402
from argopy.utils.monitored_threadpool import MyThreadPoolExecutor as MonitoredThreadPoolExecutor  # noqa: E402, F401
from argopy.utils.monitors import monitor_status as status  # noqa: E402
from argopy.related.topography import TopoFetcher
from argopy.related.ocean_ops_deployments import OceanOPSDeployments
from argopy.related.reference_tables import ArgoNVSReferenceTables
from argopy.related.argo_documentation import ArgoDocs
from argopy.related.doi_snapshot import ArgoDOI  # noqa: E402
# from argopy.extensions.canyon_med import CanyonMED  # noqa: E402
# from argopy.extensions.canyon_b import CanyonB  # noqa: E402
# from argopy.extensions.carbonate_content import CONTENT  # noqa: E402


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
    # "errors",
    # "plot",
    "ArgoColors",  # Class
    "tutorial",

    # Argo xarray accessor extensions
    # "CanyonMED",
    # "CanyonB",
    # "CONTENT",

    # Constants
    "__version__"
)

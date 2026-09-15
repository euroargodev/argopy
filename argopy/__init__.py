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

import importlib.util
import sys

def lazy_import(name, optional=False):
    """
    Lazily import a module.

    Parameters
    ----------
    name: str
        Dotted module name, e.g. "pandas" or "scipy.optimize".
    optional: bool, default=False
        If True, return None instead of raising when the module isn't installed.

    Returns
    -------
    The lazily-loaded module, or None if optional=True and the module isn't available.

    Raises
    ------
    :class:`ModuleNotFoundError` if the module isn't available and optional=False.
    """
    # Already imported (lazily or otherwise) — just return it
    if name in sys.modules:
        return sys.modules[name]

    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        # Happens when a parent package in a dotted path is missing,
        # e.g. lazy_import("foo.bar") when "foo" doesn't exist
        spec = None

    if spec is None:
        if optional:
            return None
        raise ModuleNotFoundError(f"No module named {name}")

    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module

xarray = lazy_import("xarray")
pandas = lazy_import("pandas")
erddapy = lazy_import("erddapy")
netCDF4 = lazy_import("netCDF4")
scipy = lazy_import("scipy")
IPython = lazy_import("IPython", optional=True)
pyarrow = lazy_import("pyarrow", optional=True)
seaborn = lazy_import("seaborn", optional=True)

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
from .stores import ArgoIndex, ArgoFloat, gdacfs, NVS  # noqa: E402
from .utils import show_versions, show_options  # noqa: E402
from .utils import clear_cache, lscache  # noqa: E402
from .utils import MonitoredThreadPoolExecutor  # noqa: E402, F401
from .utils import monitor_status as status  # noqa: E402
from .related import TopoFetcher, OceanOPSDeployments, ArgoDocs, ArgoDOI  # noqa: E402
from .extensions import CanyonMED  # noqa: E402
from .reference import ArgoReferenceTable, ArgoReferenceValue, ArgoReferenceMapping, ArgoNVSReferenceTables # noqa: E402
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
    "OceanOPSDeployments",  # Class
    "CTDRefDataFetcher",  # Class
    "ArgoDocs",  # Class
    "TopoFetcher",  # Class
    "ArgoDOI",  # Class

    # Argo Referencing system (vocabulary):
    "ArgoNVSReferenceTables",  # deprecated in v1.5
    "ArgoReferenceTable",
    "ArgoReferenceValue",
    "ArgoReferenceMapping",

    # Advanced Argo data stores:
    "ArgoFloat",  # Class
    "ArgoIndex",  # Class
    "gdacfs",  # Class
    "NVS", # Class

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

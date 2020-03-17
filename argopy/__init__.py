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

# Other Import
from . import utilities
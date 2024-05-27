# from .argo_index_deprec import indexstore, indexfilter_wmo, indexfilter_box
from .filesystems import filestore, httpstore, memorystore, ftpstore
from .filesystems import httpstore_erddap, httpstore_erddap_auth

from .argo_index_pa import indexstore_pyarrow as indexstore_pa
from .argo_index_pd import indexstore_pandas as indexstore_pd
from .argo_index import ArgoIndex


#
__all__ = (
    # Classes:
    "ArgoIndex",
    "indexstore_pa",
    "indexstore_pd",
    "filestore",
    "httpstore",
    "httpstore_erddap",
    "httpstore_erddap_auth",
    "ftpstore",
    "memorystore"
)

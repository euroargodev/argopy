# from .argo_index_deprec import indexstore, indexfilter_wmo, indexfilter_box
from .filesystems import filestore, httpstore, memorystore, ftpstore, s3store
from .filesystems import httpstore_erddap, httpstore_erddap_auth
from .filesystems import has_distributed, distributed  # noqa: F401

from .argo_index_pa import indexstore_pyarrow as indexstore_pa
from .argo_index_pd import indexstore_pandas as indexstore_pd

from .argo_index import ArgoIndex

from .kerchunker import ArgoKerchunker

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
    "memorystore",
    "s3store",
    "ArgoKerchunker",
)

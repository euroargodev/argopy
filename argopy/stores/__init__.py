from .filesystems import argo_store_proto
from .filesystems import has_distributed, distributed  # noqa: F401

from .implementations.local import filestore
from .implementations.memory import memorystore
from .implementations.http import httpstore
from .implementations.http_erddap import httpstore_erddap, httpstore_erddap_auth
from .implementations.ftp import ftpstore
from .implementations.s3 import s3store
from .implementations.gdac import gdacfs

from .index.argo_index import ArgoIndex
from .index.argo_index_pa import indexstore_pyarrow as indexstore_pa
from .index.argo_index_pd import indexstore_pandas as indexstore_pd

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
    "gdacfs",
)

from .spec import ArgoStoreProto
from .implementations.local import filestore
from .implementations.memory import memorystore
from .implementations.http import httpstore
from .implementations.http_erddap import httpstore_erddap, httpstore_erddap_auth
from .implementations.ftp import ftpstore
from .implementations.s3 import s3store
from .implementations.gdac import gdacfs

from .index.argo_index import ArgoIndex
from .index.implementations.index_pyarrow import indexstore as indexstore_pa
from .index.implementations.index_pandas import indexstore as indexstore_pd

from .filesystems import has_distributed, distributed  # noqa: F401
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

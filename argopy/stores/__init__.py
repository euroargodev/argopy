from .implementations.local import filestore
from .implementations.memory import memorystore
from .implementations.http import httpstore
from .implementations.http_erddap import httpstore_erddap
from .implementations.ftp import ftpstore
from .implementations.s3 import s3store
from .implementations.gdac import gdacfs

from .index.argo_index import ArgoIndex
from .float.argo_float import ArgoFloat

from .kerchunker import ArgoKerchunker

from .filesystems import has_distributed, distributed  # noqa: F401
from .spec import ArgoStoreProto  # noqa: F401
from .implementations.http_erddap import httpstore_erddap_auth  # noqa: F401


__all__ = (
    # Classes:
    "ArgoIndex",
    "ArgoFloat",
    "filestore",
    "httpstore",
    "httpstore_erddap",
    "ftpstore",
    "memorystore",
    "s3store",
    "ArgoKerchunker",
    "gdacfs",
)

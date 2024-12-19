from typing import Union
from pathlib import Path
from fsspec.core import split_protocol
from urllib.parse import urlparse
from socket import gaierror

from ...options import OPTIONS
from ...errors import GdacPathError
from .. import filestore, httpstore, ftpstore, s3store


class gdacfs:
    """Argo file system for any GDAC path

    Parameters
    ----------
    path: str, optional
        GDAC path to create a file system for. Support any possible GDAC path.
        If not specified, value from global option ``gdac`` will be used.

    Returns
    -------
    A file system based on :class:`argopy.stores.ArgoStoreProto`

    Examples
    --------

    >>> fs = gdacfs("https://data-argo.ifremer.fr")
    >>> fs = gdacfs("https://usgodae.org/pub/outgoing/argo")
    >>> fs = gdacfs("ftp://ftp.ifremer.fr/ifremer/argo")
    >>> fs = gdacfs("/home/ref-argo/gdac")
    >>> fs = gdacfs("s3://argo-gdac-sandbox/pub")

    >>> with argopy.set_options(gdac="s3://argo-gdac-sandbox/pub"):
    >>>     fs = gdacfs()

    Warnings
    --------
    This class does not check if the path is a valid Argo GDAC

    See Also
    --------
    :meth:`argopy.utils.check_gdac_path`, :meth:`argopy.utils.list_gdac_servers`

    """
    protocol2fs = {"file": filestore, "http": httpstore, "ftp": ftpstore, "s3": s3store}
    """Dictionary mapping path protocol to Argo file system to instantiate"""

    @staticmethod
    def path2protocol(path: Union[str, Path]) -> str:
        """Narrow down any path to a supported protocols, raise GdacPathError if protocol not supported"""
        if isinstance(path, Path):
            return "file"
        else:
            split = split_protocol(path)[0]
            if split is None:
                return "file"
            if "http" in split:  # will also catch "https"
                return "http"
            elif "ftp" in split:
                return "ftp"
            elif "s3" in split:
                return "s3"
            else:
                raise GdacPathError("Unknown protocol for an Argo GDAC host: %s" % split)

    def __new__(cls, path: Union[str, Path, None] = None):
        """Create a file system for any Argo GDAC compliant path"""
        if path is None:
            path = OPTIONS["gdac"]

        protocol = cls.path2protocol(path)
        fs = cls.protocol2fs[protocol]
        fs_args = {}

        if protocol == "ftp":
            ftp_host = urlparse(path).hostname
            ftp_port = 0 if urlparse(path).port is None else urlparse(path).port
            fs_args['host'] = ftp_host
            fs_args['port'] = ftp_port

        try:
            fs = fs(**fs_args)
        except gaierror as e:
            raise GdacPathError(
                "Can't get address info from FTP host: %s\nGAIerror: %s"
                % (fs_args, str(e))
            )
        return fs

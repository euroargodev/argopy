import os
import fsspec
import logging
import importlib
from typing import Union

from ..options import OPTIONS
from ..utils.accessories import Registry
from .. import __version__


if importlib.util.find_spec("boto3") is not None:
    HAS_BOTO3 = True
    import boto3
else:
    HAS_BOTO3 = False


log = logging.getLogger("argopy.stores")


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    log.debug("argopy needs 'tqdm' to display progress bars")

    def tqdm(fct, **kw):
        return fct


try:
    import distributed

    has_distributed = True
except ModuleNotFoundError:
    log.debug("argopy needs 'distributed' to use Dask cluster/client")
    has_distributed = False
    distributed = None


def new_fs(
    protocol: str = "",
    cache: bool = False,
    cachedir: str = OPTIONS["cachedir"],
    cache_expiration: int = OPTIONS["cache_expiration"],
    **kwargs,
) -> (fsspec.spec.AbstractFileSystem, Union[Registry, None]):
    """Create a new fsspec file system for argopy higher level stores

    Parameters
    ----------
    protocol: str (optional)
    cache: bool (optional)
        Use a filecache system on top of the protocol. Default: False
    cachedir: str
        Define path to cache directory.
    **kwargs: (optional)
        Other arguments passed to :func:`fsspec.filesystem`

    Returns
    -------
    (fs, cache_registry)
        A tuple with the fsspec file system and :class:`argopy.Registry` for cache if any

    """
    # Merge default FSSPEC kwargs with user defined kwargs:
    default_fsspec_kwargs = {"simple_links": True, "block_size": 0}
    if protocol == "http":
        client_kwargs = {
            "trust_env": OPTIONS["trust_env"],
            "headers": {"Argopy-Version": __version__},
        }  # Passed to aiohttp.ClientSession
        if "client_kwargs" in kwargs:
            client_kwargs = {**client_kwargs, **kwargs["client_kwargs"]}
            kwargs.pop("client_kwargs")
        default_fsspec_kwargs = {
            **default_fsspec_kwargs,
            **{"client_kwargs": {**client_kwargs}},
        }
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    elif protocol == "ftp":
        default_fsspec_kwargs = {
            **default_fsspec_kwargs,
            **{"block_size": 1000 * (2**20)},
        }
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    elif protocol == "s3":
        default_fsspec_kwargs.pop("simple_links")
        default_fsspec_kwargs.pop("block_size")
        if "anon" not in kwargs:
            default_fsspec_kwargs["anon"] = (
                boto3.client("s3")._request_signer._credentials is None
                if HAS_BOTO3
                else True
            )
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    else:
        # Merge default with user arguments:
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    # Create filesystem:
    if not cache:
        fs = fsspec.filesystem(protocol, **fsspec_kwargs)
        cache_registry = None
        log_msg = (
            "Opening a fsspec [file] system for '%s' protocol with options: %s"
            % (protocol, str(fsspec_kwargs))
        )
    else:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fs = fsspec.filesystem(
            "filecache",
            target_protocol=protocol,
            target_options={**fsspec_kwargs},
            cache_storage=cachedir,
            expiry_time=cache_expiration,
            cache_check=10,
        )

        cache_registry = Registry(
            name="Cache"
        )  # Will hold uri cached by this store instance
        log_msg = (
            "Opening a fsspec [filecache, storage='%s'] system for '%s' protocol with options: %s"
            % (cachedir, protocol, str(fsspec_kwargs))
        )

    if (
        protocol == "file"
        and os.path.sep != fs.sep
        # and version.parse(fsspec.__version__) < version.parse("2025.3.0")
        # and os.name == "nt"
    ):
        # For some reason (see https://github.com/fsspec/filesystem_spec/issues/937), the property fs.sep is
        # not '\' under Windows. So, using this dirty fix to overwrite it:
        log.debug("Found os.path.sep ('%s') != fs.sep ('%s')" % (os.path.sep, fs.sep))
        fs.sep = os.path.sep
        # fsspec folks recommend to use posix internally. But I don't see how to handle this. So keeping this fix
        # because it solves issues with failing tests under Windows. Enough at this time.

    # log_msg = "%s\n[sys sep=%s] vs [fs sep=%s]" % (log_msg, os.path.sep, fs.sep)
    # log.warning(log_msg)
    log.debug(log_msg)
    # log_argopy_callerstack()
    return fs, cache_registry, fsspec_kwargs

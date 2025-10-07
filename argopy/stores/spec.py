from abc import ABC, abstractmethod
import fsspec
from packaging import version
import os
import shutil
import pickle  # nosec B403 only used with internal files/assets
import json
import tempfile
import aiohttp
from typing import Union
from pathlib import Path
import logging


from ..options import OPTIONS
from ..errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
)
from .filesystems import new_fs


log = logging.getLogger("argopy.stores.spec")


class ArgoStoreProto(ABC):
    """Argo File System Prototype

    All argopy file systems must inherit, directly or not, from this prototype.

    Should this class inherits from :class:`fsspec.spec.AbstractFileSystem` ?
    """

    protocol = ""
    """str: File system name, one in :class:`fsspec.registry.known_implementations`"""

    def __init__(self, cache: bool = False, cachedir: str = "", **kwargs):
        """Create a file storage system for Argo data

        Parameters
        ----------
        cache: bool (False)
        cachedir: str (from OPTIONS)
        **kwargs: (optional)
            Other arguments are passed to :func:`fsspec.filesystem`

        """
        self.cache = cache
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self._fsspec_kwargs = {**kwargs}
        self.fs, self.cache_registry, self._fsspec_kwargs = new_fs(
            self.protocol, self.cache, self.cachedir, **self._fsspec_kwargs
        )

    def open(self, path, *args, **kwargs):
        self.register(path)
        # log.debug("Opening path: %s" % path)
        return self.fs.open(path, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def ls(self, path, **kwargs):
        return self.fs.ls(path, **kwargs)

    @property
    def sep(self):
        return self.fs.sep

    @property
    def async_impl(self):
        return self.fs.async_impl

    @property
    def asynchronous(self):
        return getattr(self.fs, 'asynchronous', False)

    @property
    def target_protocol(self):
        return getattr(self.fs, 'target_protocol', self.protocol)

    def unstrip_protocol(self, path, **kwargs):
        return self.fs.unstrip_protocol(path, **kwargs)

    def exists(self, path, *args):
        return self.fs.exists(path, *args)

    def info(self, path, *args, **kwargs):
        if self.fs.protocol == "dir":
            info = self.fs.fs.info(self.fs._join(path), **kwargs)
            info = info.copy()
            # info["name"] = self.fs._relpath(info["name"])  # Raw code from fsspec
            info["name"] = self.fs._relpath(self.fs._join(path))  # Fix https://github.com/euroargodev/argopy/issues/499
            return info
        else:
            return self.fs.info(path, *args, **kwargs)

    def first(self, path: Union[str, Path], N: int = 4) -> str:
        """Read first N bytes of a path

        Return None if path cannot be open

        Parameters
        ----------
        path: str, Path

        Raises
        ------
        :class:`aiohttp.ClientResponseError`
        """
        def is_read(uri):
            try:
                self.ls(uri)
                return True
            except aiohttp.ClientResponseError:
                raise
            except Exception:
                return False

        if is_read(str(path)):
            try:
                return self.fs.open(str(path)).read(N)
            except:  # noqa: E722
                return None
        else:
            return None

    def expand_path(self, path, **kwargs):
        """Turn one or more globs or directories into a list of all matching paths to files or directories.

        For http store, return path unchanged (not implemented).

        kwargs are passed to fsspec expand_path which call ``glob`` or ``find``, which may in turn call ``ls``.

        Returns
        -------
        list
        """
        if self.protocol != "http" and self.protocol != "https":
            return self.fs.expand_path(path, **kwargs)
        else:
            return [path]

    def store_path(self, uri):
        path = uri
        path = self.expand_path(path)[0]
        if not path.startswith(self.target_protocol) and version.parse(
            fsspec.__version__
        ) <= version.parse("0.8.3"):
            path = self.fs.target_protocol + "://" + path
        return path

    def full_path(self, path, protocol: bool = False):
        """Return fully developed path

        Examples
        --------
        full_path('')

        """
        fp = getattr(self.fs, '_join', lambda x: x)(path)
        if self.protocol == 'ftp':
            fp = f"{self.host}:{self.port}{self.fs._strip_protocol(fp)}"
        if not protocol:
            return fp
        else:
            if self.fs.protocol == "dir":
                return self.fs.fs.unstrip_protocol(fp)
            else:
                return self.unstrip_protocol(fp)

    def register(self, uri):
        """Keep track of files open with this instance"""
        if self.cache:
            path = self.store_path(uri)
            if path not in self.cache_registry:
                self.cache_registry.commit(path)

    @property
    def cached_files(self):
        # See https://github.com/euroargodev/argopy/issues/294
        if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
            return self.fs.cached_files
        else:
            return self.fs._metadata.cached_files

    def cachepath(self, uri: str, errors: str = "raise"):
        """Return path to cached file for a given URI"""
        if not self.cache:
            if errors == "raise":
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        elif uri is not None:
            store_path = self.store_path(uri)
            self.fs.load_cache()  # Read set of stored blocks from file and populate self.fs.cached_files
            if store_path in self.cached_files[-1]:
                return os.path.sep.join(
                    [self.cachedir, self.cached_files[-1][store_path]["fn"]]
                )
            elif errors == "raise":
                raise CacheFileNotFound(
                    "No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri)
                )
        else:
            raise CacheFileNotFound(
                "No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri)
            )

    def _clear_cache_item(self, uri):
        """Remove metadata and file for fsspec cache uri"""
        fn = os.path.join(self.fs.storage[-1], "cache")
        self.fs.load_cache()  # Read set of stored blocks from file and populate self.cached_files
        cache = self.cached_files[-1]

        # Read cache metadata:
        if os.path.exists(fn):
            if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
                with open(fn, "rb") as f:
                    cached_files = pickle.load(
                        f
                    )  # nosec B301 because files controlled internally
            else:
                with open(fn, "r") as f:
                    cached_files = json.load(f)
        else:
            cached_files = cache

        # Build new metadata without uri to delete, and delete corresponding cached file:
        cache = {}
        for k, v in cached_files.items():
            if k != uri:
                cache[k] = v.copy()
            else:
                # Delete file:
                os.remove(os.path.join(self.fs.storage[-1], v["fn"]))
                # log.debug("Removed %s -> %s" % (uri, v['fn']))

        # Update cache metadata file:
        if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                pickle.dump(cache, f)
            shutil.move(f.name, fn)
        else:
            with fsspec.utils.atomic_write(fn, mode="w") as f:
                json.dump(cache, f)

    def clear_cache(self):
        """Remove cache files and entry from uri open with this store instance"""
        if self.cache:
            for uri in self.cache_registry:
                # log.debug("Removing from cache %s" % uri)
                self._clear_cache_item(uri)
            self.cache_registry.clear()  # Reset registry

    @abstractmethod
    def open_dataset(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_csv(self):
        raise NotImplementedError("Not implemented")

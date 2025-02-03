from .local import filestore
from ...errors import CacheFileNotFound, FileSystemHasNoCache


class memorystore(filestore):
    """Argo in-memory file system (global)

    Note that this inherits from :class:`argopy.stores.filestore`, not the:class:`argopy.stores.ArgoStoreProto`.

    Relies on :class:`fsspec.implementations.memory.MemoryFileSystem`
    """

    protocol = "memory"

    def exists(self, path, *args):
        """Check if path can be open or not

        Special handling for memory store

        The fsspec.exists() will return False even if the path is in cache.
        Here we bypass this in order to return True if the path is in cache.
        This assumes that the goal of fs.exists is to determine if we can load the path or not.
        If the path is in cache, it can be loaded.
        """
        guess = self.fs.exists(path, *args)
        if not guess:
            try:
                self.cachepath(path)
                return True
            except CacheFileNotFound:
                pass
            except FileSystemHasNoCache:
                pass
        return guess

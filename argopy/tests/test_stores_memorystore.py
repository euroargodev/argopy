import pytest
import fsspec
import logging
from argopy.stores import memorystore
from argopy.errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
)


log = logging.getLogger("argopy.tests.stores")


class Test_MemoryStore:

    def test_implementation(self):
        fs = memorystore(cache=False)
        assert isinstance(fs.fs, fsspec.implementations.memory.MemoryFileSystem)

    def test_nocache(self):
        fs = memorystore(cache=False)
        with pytest.raises(FileSystemHasNoCache):
            fs.cachepath("dummy_uri")

    def test_cacheable(self):
        fs = memorystore(cache=True)
        assert isinstance(fs.fs, fsspec.implementations.cached.WholeFileCacheFileSystem)

    def test_nocachefile(self):
        fs = memorystore(cache=True)
        with pytest.raises(CacheFileNotFound):
            fs.cachepath("dummy_uri")

    def test_exists(self):
        fs = memorystore(cache=False)
        assert not fs.exists('dummy.txt')
        fs = memorystore(cache=True)
        assert not fs.exists('dummy.txt')

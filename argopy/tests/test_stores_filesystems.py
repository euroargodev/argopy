import pytest
from fsspec.registry import known_implementations
import importlib
import logging

from argopy.stores.filesystems import new_fs


log = logging.getLogger("argopy.tests.stores")


id_implementation = lambda x: [k for k, v in known_implementations.items()  # noqa: E731
                                      if x.__class__.__name__ == v['class'].split('.')[-1]]
is_initialised = lambda x: ((x is None) or (x == []))  # noqa: E731


class Test_new_fs:

    def test_default(self):
        fs, cache_registry, fsspec_kwargs = new_fs()
        assert id_implementation(fs) is not None
        assert is_initialised(cache_registry)

    def test_cache_type(self):
        fs, cache_registry, fsspec_kwargs = new_fs(cache=True)
        assert id_implementation(fs) == ['filecache']

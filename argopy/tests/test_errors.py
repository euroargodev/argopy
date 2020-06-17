import pytest
from argopy.errors import NetCDF4FileNotFoundError, \
    CacheFileNotFound, \
    FileSystemHasNoCache, \
    UnrecognisedDataSelectionMode, \
    UnrecognisedProfileDirection, \
    InvalidDatasetStructure, \
    InvalidFetcherAccessPoint, \
    InvalidFetcher


def test_NetCDF4FileNotFoundError():
    def foobar():
        raise NetCDF4FileNotFoundError("invalid_path")
    with pytest.raises(NetCDF4FileNotFoundError):
        foobar()


def test_CacheFileNotFound():
    def foobar():
        raise CacheFileNotFound()
    with pytest.raises(CacheFileNotFound):
        foobar()


def test_FileSystemHasNoCache():
    def foobar():
        raise FileSystemHasNoCache()
    with pytest.raises(FileSystemHasNoCache):
        foobar()


def test_UnrecognisedDataSelectionMode():
    def foobar():
        raise UnrecognisedDataSelectionMode()
    with pytest.raises(UnrecognisedDataSelectionMode):
        foobar()


def test_UnrecognisedProfileDirection():
    def foobar():
        raise UnrecognisedProfileDirection()
    with pytest.raises(UnrecognisedProfileDirection):
        foobar()


def test_InvalidDatasetStructure():
    def foobar():
        raise InvalidDatasetStructure()
    with pytest.raises(InvalidDatasetStructure):
        foobar()


def test_InvalidFetcherAccessPoint():
    def foobar():
        raise InvalidFetcherAccessPoint()
    with pytest.raises(InvalidFetcherAccessPoint):
        foobar()


def test_InvalidFetcher():
    def foobar():
        raise InvalidFetcher()
    with pytest.raises(InvalidFetcher):
        foobar()

import pytest
from argopy.errors import (
    DataNotFound,
    FtpPathError,
    NetCDF4FileNotFoundError,
    CacheFileNotFound,
    FileSystemHasNoCache,
    UnrecognisedDataSelectionMode,
    UnrecognisedProfileDirection,
    InvalidDatasetStructure,
    InvalidFetcherAccessPoint,
    InvalidFetcher,
    InvalidMethod,
    InvalidDashboard,
    APIServerError,
    ErddapServerError,
    ArgovisServerError
)


class Test_Errors:
    def __test_one(self, ThisError):
        with pytest.raises(ThisError):
            raise ThisError()

    def test_RaiseAll(self):
        elist = [
            DataNotFound,
            FtpPathError,
            NetCDF4FileNotFoundError,
            CacheFileNotFound,
            FileSystemHasNoCache,
            UnrecognisedDataSelectionMode,
            UnrecognisedProfileDirection,
            InvalidDatasetStructure,
            InvalidFetcherAccessPoint,
            InvalidFetcher,
            InvalidMethod,
            InvalidDashboard,
            APIServerError,
            ErddapServerError,
            ArgovisServerError
        ]
        for e in elist:
            self.__test_one(e)

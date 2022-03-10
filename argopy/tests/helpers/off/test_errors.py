import pytest
from argopy.errors import (
    DataNotFound,
    FtpPathError,
    NetCDF4FileNotFoundError,
    CacheFileNotFound,
    FileSystemHasNoCache,
    UnrecognisedProfileDirection,
    InvalidDataset,
    InvalidDatasetStructure,
    InvalidFetcherAccessPoint,
    InvalidFetcher,
    InvalidOption,
    OptionValueError,
    InvalidMethod,
    InvalidDashboard,
    APIServerError,
    ErddapServerError,
    ArgovisServerError
)


@pytest.mark.parametrize("error", [
    DataNotFound,
    FtpPathError,
    NetCDF4FileNotFoundError,
    CacheFileNotFound,
    FileSystemHasNoCache,
    UnrecognisedProfileDirection,
    InvalidDataset,
    InvalidDatasetStructure,
    InvalidFetcherAccessPoint,
    InvalidFetcher,
    InvalidOption,
    OptionValueError,
    InvalidMethod,
    InvalidDashboard,
    APIServerError,
    ErddapServerError,
    ArgovisServerError,
    ], indirect=False)
def test_raise_all_errors(error):
    with pytest.raises(error):
        raise error()

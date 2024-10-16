"""A bunch of custom errors used in argopy."""
from typing import List
import warnings
import logging

log = logging.getLogger("argopy.errors")


class NoData(ValueError):
    """Raise for no data"""
    def __init__(self, path: str = "?"):
        self.value = "%s" % path
        self.path = path

    def __str__(self):
        """Print error."""
        return repr(self.value)


class DataNotFound(NoData):
    """Raise when a data fetching returns nothing"""
    pass


class NoDataLeft(NoData):
    """Raise when data processing returns an empty dataset or dataframe"""
    pass


class GdacPathError(ValueError):
    """Raise when a GDAC compliant path is not appropriate"""

    pass


class ErddapPathError(ValueError):
    """Raise when an erddap path is not appropriate"""

    pass


class S3PathError(ValueError):
    """Raise when a S3 path is not appropriate"""

    pass


class NetCDF4FileNotFoundError(FileNotFoundError):
    """Raise when NETCDF4 file not found."""

    def __init__(self, path: str = "?"):
        self.value = "Couldn't find NetCDF4 file: %s" % path
        self.path = path

    def __str__(self):
        """Print error."""
        return repr(self.value)


class CacheFileNotFound(FileNotFoundError):
    """Raise when a file is not found in cache."""

    pass


class FileSystemHasNoCache(ValueError):
    """Raise when trying to access a cache system not implemented."""

    pass


class UnrecognisedProfileDirection(ValueError):
    """Not "A" or "D". Argopy should have recognized those."""

    def __init__(self, institute=None, wmo=None):
        self.institute = institute
        self.wmo = wmo


class InvalidDataset(ValueError):
    """
    This is to be used when a dataset or its property is not valid
    """
    pass


class InvalidDatasetStructure(ValueError):
    """Raise when the xarray dataset is not as expected."""

    pass


class InvalidFetcherAccessPoint(ValueError):
    """Raise when requesting a fetcher access point not available."""

    pass


class InvalidFetcher(ValueError):
    """Raise when trying to do something with a fetcher not ready."""

    pass


class InvalidOption(ValueError):
    """Raise when trying to set an invalid option name."""

    pass


class OptionValueError(ValueError):
    """Raise when the option value is not valid."""

    pass


class InvalidMethod(ValueError):
    """Raise when trying to do use a method not available."""

    pass


class InvalidDashboard(ValueError):
    """Raise this when trying to work with a 3rd party online service to display float information."""

    pass


class APIServerError(ValueError):
    """Raise this when argopy is disrupted by an error due to a webAPI, not argopy machinery."""
    def __init__(self, path: str = None):
        self.value = path

    def __str__(self):
        """Print error."""
        return repr(self.value)


class ErddapServerError(APIServerError):
    """Raise this when argopy is disrupted by an error due to the Erddap server, not argopy machinery."""

    pass


class ArgovisServerError(APIServerError):
    """Raise this when argopy is disrupted by an error due to the Argovis server, not argopy machinery."""

    pass


class ErddapHTTPUnauthorized(APIServerError):
    """Raise when login to erddap fails"""

    pass


class ErddapHTTPNotFound(APIServerError):
    """Raise when erddap resource is not found"""

    pass


class OptionDeprecatedWarning(DeprecationWarning):
    """When an option being deprecated is used

    This is a class to emit a warning when an option being deprecated is used.

    Parameters
    ----------
    reason: str, optional, default=None
        Text message to send with deprecation warning
    version: str, optional, default=None
    ignore_caller: List, optional, default=[]
    """
    def __init__(self, reason: str = None, version: str = None, ignore_caller: List = []):
        import inspect
        ignore_caller = [ignore_caller]

        if isinstance(reason, str):

            fmt = "\nCall to deprecated option: {reason}"
            if version is not None:
                fmt = "%s -- Deprecated since version {version}" % fmt

            issue_deprec = True
            stack = inspect.stack()
            for s in stack:
                if "<module>" in s.function:
                    break
                elif s.function in ignore_caller:
                    issue_deprec = False

            if issue_deprec:
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt.format(reason=reason, version=version),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
            else:
                log.warning(fmt.format(reason=reason, version=version))

        else:
            raise TypeError(repr(type(reason)))

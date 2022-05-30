"""A bunch of custom errors used in argopy."""


class DataNotFound(ValueError):
    """Raise when a data selection returns nothing."""

    def __init__(self, path: str = "?"):
        self.value = "%s" % path
        self.path = path

    def __str__(self):
        """Print error."""
        return repr(self.value)


class FtpPathError(ValueError):
    """Raise when the ftp path is not appropriate."""

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
    """Raise this when argopy is disrupted by an error due to the Erddap, not argopy machinery."""

    pass


class ArgovisServerError(APIServerError):
    """Raise this when argopy is disrupted by an error due to the Erddap, not argopy machinery."""

    pass

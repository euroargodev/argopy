# Legacy author (2019) sean.tokunaga@ifremer.fr

"""
A bunch of custom errors used in argopy.
"""

class DataNotFound(ValueError):
    """ Raise when a data selection returns nothing """
    pass

class FtpPathError(ValueError):
    """ Raise when the ftp path is not appropriate """
    pass


class NetCDF4FileNotFoundError(FileNotFoundError):
    """
    Most common error. Basically just a file not found.
    I made a custom one to make it easier to catch.
    """
    def __init__(self, path):
        self.value = "Couldn't find NetCDF4 file: %s" % path
        self.path = path

    def __str__(self):
        return (repr(self.value))


class CacheFileNotFound(FileNotFoundError):
    """ Raise when a file is not found in cache """
    pass


class FileSystemHasNoCache(ValueError):
    """ Raise when trying to access a cache system not implemented """
    pass


class UnrecognisedDataSelectionMode(ValueError):
    """
    This is to be used when the user fails to specify a valid data selection mode.
    the valid ones to date are ""delayed_mode", "real-time", "adj_non_empty" (default), "data_mode"
    """
    def __init__(self, institute=None, wmo=None):
        self.institute = institute
        self.wmo = wmo


class UnrecognisedProfileDirection(ValueError):
    """
    Not "A" or "D". Argopy should have recognized those.
    """
    def __init__(self, institute=None, wmo=None):
        self.institute = institute
        self.wmo = wmo


class InvalidDatasetStructure(ValueError):
    """
    This is to be used when the in-memory xarray dataset is not structured as expected
    """
    pass


class InvalidFetcherAccessPoint(ValueError):
    """
    Raise when requesting a fetcher access point not available
    """
    pass


class InvalidFetcher(ValueError):
    """
    Raise when trying to do something with a fetcher not ready
    """
    pass


class ErddapServerError(ValueError):
    """
    Raise this when argopy is disrupted by an error due to the Erddap, not argopy machinery
    """
    pass


class InvalidDashboard(ValueError):
    """
    Raise this when trying to work with a 3rd party online service to display float information
    """
    pass

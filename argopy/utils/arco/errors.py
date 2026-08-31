__author__ = 'sean.tokunaga@ifremer.fr'

"""
A bunch of custom errors used in argopy.
"""

class NetCDF4FileNotFoundError(FileNotFoundError):
    """
    Most common error. Basically just a file not found.
    I made a custom one to make it easier to catch.
    """
    def __init__(self, path, verbose=True):
        if verbose:
            print("Error: Couldn't find NetCDF4 file:")
            print(path)
        self.path = path


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

class NCParsed(object):
    """
    Abstract object for parsed .nc objects.

    The main purpose for having this base class is to be able to store flags that get raised,
    as well as a standardized place for storing the variable names in a xarray file.
    """
    def __init__(self, arr, meta=None):
        if meta and type(meta) is dict:
            self.__dict__.update(meta)
        self.data_vars = list(arr.data_vars)
        self.flags = []

    def verify(self, *argv):
        pass

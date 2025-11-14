import warnings
from argopy.utils.checkers import to_list
from argopy.utils.accessories import RegistryItem


def check_wmo(lst, errors="raise"):
    """Validate a WMO option and returned it as a list of integers

    Parameters
    ----------
    wmo: int
        WMO must be an integer or an iterable with elements that can be casted as integers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

    Returns
    -------
    list(int)
    """
    is_wmo(lst, errors=errors)

    # Make sure we deal with a list
    lst = to_list(lst)

    # Then cast list elements as integers
    return [abs(int(x)) for x in lst]


def is_wmo(lst, errors="raise"):  # noqa: C901
    """Check if a WMO is valid

    Parameters
    ----------
    wmo: int, list(int), array(int)
        WMO must be a single or a list of 5/7 digit positive numbers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

    Returns
    -------
    bool
        True if wmo is indeed a list of integers
    """

    # Make sure we deal with a list
    lst = to_list(lst)

    # Error message:
    # msg = "WMO must be an integer or an iterable with elements that can be casted as integers"
    msg = "WMO must be a single or a list of 5/7 digit positive numbers. Invalid: '{}'".format

    # Then try to cast list elements as integers, return True if ok
    result = True
    try:
        for x in lst:
            if not str(x).isdigit():
                result = False

            if (len(str(x)) != 5) and (len(str(x)) != 7):
                result = False

            if int(x) <= 0:
                result = False

    except Exception:
        result = False
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == "warn":
            warnings.warn(msg(x))

    if not result:
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == "warn":
            warnings.warn(msg(x))
    else:
        return result


class float_wmo(RegistryItem):
    """Argo float WMO number object"""

    def __init__(self, WMO_number, errors="raise"):
        """Create an Argo float WMO number object

        Parameters
        ----------
        WMO_number: object
            Anything that could be cast as an integer
        errors: {'raise', 'warn', 'ignore'}
            Possibly raises a ValueError exception or UserWarning, otherwise fails silently if WMO_number is not valid

        Returns
        -------
        obj
        """
        self.errors = errors
        if isinstance(WMO_number, float_wmo):
            item = WMO_number.value
        else:
            item = check_wmo(WMO_number, errors=self.errors)[
                0
            ]  # This will automatically validate item
        self.item = item

    @property
    def isvalid(self):
        """Check if WMO number is valid"""
        return is_wmo(self.item, errors=self.errors)
        # return True  # Because it was checked at instantiation

    @property
    def value(self):
        """Return WMO number as in integer"""
        return int(self.item)

    def __str__(self):
        # return "%s" % check_wmo(self.item)[0]
        return "%s" % self.item

    def __repr__(self):
        return f"WMO({self.item})"

    def __check_other__(self, other):
        return check_wmo(other)[0] if type(other) is not float_wmo else other.item

    def __eq__(self, other):
        return self.item.__eq__(self.__check_other__(other))

    def __ne__(self, other):
        return self.item.__ne__(self.__check_other__(other))

    def __gt__(self, other):
        return self.item.__gt__(self.__check_other__(other))

    def __lt__(self, other):
        return self.item.__lt__(self.__check_other__(other))

    def __ge__(self, other):
        return self.item.__ge__(self.__check_other__(other))

    def __le__(self, other):
        return self.item.__le__(self.__check_other__(other))

    def __hash__(self):
        return hash(self.item)

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from ..utilities import to_list
from ..errors import InvalidDatasetStructure


def is_indexbox(box: list, errors="raise"):
    """ Check if this array matches a 2d or 3d index box definition

    Argopy expects one of the following 2 format to define an index box:

    - box = [lon_min, lon_max, lat_min, lat_max]
    - box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]

    This function check for this format compliance.

    Parameters
    ----------
    box: list
    errors: str, default='raise'

    Returns
    -------
    bool
    """
    def is_dateconvertible(d):
        try:
            pd.to_datetime(d)
            isit = True
        except Exception:
            isit = False
        return isit

    tests = {}

    # Formats:
    tests["index box must be a list"] = lambda b: isinstance(b, list)
    tests["index box must be a list with 4 or 6 elements"] = lambda b: len(b) in [4, 6]

    # Types:
    tests["lon_min must be numeric"] = lambda b: (
        isinstance(b[0], int) or isinstance(b[0], (np.floating, float))
    )
    tests["lon_max must be numeric"] = lambda b: (
        isinstance(b[1], int) or isinstance(b[1], (np.floating, float))
    )
    tests["lat_min must be numeric"] = lambda b: (
        isinstance(b[2], int) or isinstance(b[2], (np.floating, float))
    )
    tests["lat_max must be numeric"] = lambda b: (
        isinstance(b[3], int) or isinstance(b[3], (np.floating, float))
    )
    if len(box) > 4:
        tests[
            "datetim_min must be a string convertible to a Pandas datetime"
        ] = lambda b: isinstance(b[-2], str) and is_dateconvertible(b[-2])
        tests[
            "datetim_max must be a string convertible to a Pandas datetime"
        ] = lambda b: isinstance(b[-1], str) and is_dateconvertible(b[-1])

    # Ranges:
    tests["lon_min must be in [-180;180] or [0;360]"] = (
        lambda b: b[0] >= -180.0 and b[0] <= 360.0
    )
    tests["lon_max must be in [-180;180] or [0;360]"] = (
        lambda b: b[1] >= -180.0 and b[1] <= 360.0
    )
    tests["lat_min must be in [-90;90]"] = lambda b: b[2] >= -90.0 and b[2] <= 90
    tests["lat_max must be in [-90;90]"] = lambda b: b[3] >= -90.0 and b[3] <= 90.0

    # Orders:
    tests["lon_max must be larger than lon_min"] = lambda b: b[0] < b[1]
    tests["lat_max must be larger than lat_min"] = lambda b: b[2] < b[3]
    if len(box) > 4:
        tests["datetim_max must come after datetim_min"] = lambda b: pd.to_datetime(
            b[-2]
        ) < pd.to_datetime(b[-1])

    error = None
    for msg, test in tests.items():
        if not test(box):
            error = msg
            break

    if error and errors == "raise":
        raise ValueError("%s: %s" % (box, error))
    elif error:
        return False
    else:
        return True


def is_box(box: list, errors="raise"):
    """Check if this array matches a 3d or 4d data box definition

    Argopy expects one of the following 2 format to define a box:

    - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
    - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]

    This function check for this format compliance.

    Parameters
    ----------
    box: list
    errors: 'raise'

    Returns
    -------
    bool
    """

    def is_dateconvertible(d):
        try:
            pd.to_datetime(d)
            isit = True
        except Exception:
            isit = False
        return isit

    tests = {}
    #     print(box)
    # Formats:
    tests["box must be a list"] = lambda b: isinstance(b, list)
    tests["box must be a list with 6 or 8 elements"] = lambda b: len(b) in [6, 8]

    # Types:
    tests["lon_min must be numeric"] = lambda b: (
        isinstance(b[0], int) or isinstance(b[0], (np.floating, float))
    )
    tests["lon_max must be numeric"] = lambda b: (
        isinstance(b[1], int) or isinstance(b[1], (np.floating, float))
    )
    tests["lat_min must be numeric"] = lambda b: (
        isinstance(b[2], int) or isinstance(b[2], (np.floating, float))
    )
    tests["lat_max must be numeric"] = lambda b: (
        isinstance(b[3], int) or isinstance(b[3], (np.floating, float))
    )
    tests["pres_min must be numeric"] = lambda b: (
        isinstance(b[4], int) or isinstance(b[4], (np.floating, float))
    )
    tests["pres_max must be numeric"] = lambda b: (
        isinstance(b[5], int) or isinstance(b[5], (np.floating, float))
    )
    if len(box) == 8:
        tests[
            "datetim_min must be an object convertible to a Pandas datetime"
        ] = lambda b: is_dateconvertible(b[-2])
        tests[
            "datetim_max must be an object convertible to a Pandas datetime"
        ] = lambda b: is_dateconvertible(b[-1])

    # Ranges:
    tests["lon_min must be in [-180;180] or [0;360]"] = (
        lambda b: b[0] >= -180.0 and b[0] <= 360.0
    )
    tests["lon_max must be in [-180;180] or [0;360]"] = (
        lambda b: b[1] >= -180.0 and b[1] <= 360.0
    )
    tests["lat_min must be in [-90;90]"] = lambda b: b[2] >= -90.0 and b[2] <= 90
    tests["lat_max must be in [-90;90]"] = lambda b: b[3] >= -90.0 and b[3] <= 90.0
    tests["pres_min must be in [0;10000]"] = lambda b: b[4] >= 0 and b[4] <= 10000
    tests["pres_max must be in [0;10000]"] = lambda b: b[5] >= 0 and b[5] <= 10000

    # Orders:
    tests["lon_max must be larger than lon_min"] = lambda b: b[0] <= b[1]
    tests["lat_max must be larger than lat_min"] = lambda b: b[2] <= b[3]
    tests["pres_max must be larger than pres_min"] = lambda b: b[4] <= b[5]
    if len(box) == 8:
        tests["datetim_max must come after datetim_min"] = lambda b: pd.to_datetime(
            b[-2]
        ) <= pd.to_datetime(b[-1])

    error = None
    for msg, test in tests.items():
        if not test(box):
            error = msg
            break

    if error and errors == "raise":
        raise ValueError("%s: %s" % (box, error))
    elif error:
        return False
    else:
        return True


def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def is_list_of_dicts(lst):
    return all(isinstance(x, dict) for x in lst)


def is_list_of_datasets(lst):
    return all(isinstance(x, xr.Dataset) for x in lst)


def is_list_equal(lst1, lst2):
    """ Return true if 2 lists contain same elements"""
    return len(lst1) == len(lst2) and len(lst1) == sum(
        [1 for i, j in zip(lst1, lst2) if i == j]
    )


def check_wmo(lst, errors="raise"):
    """ Validate a WMO option and returned it as a list of integers

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
    """ Check if a WMO is valid

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
        elif errors == 'warn':
            warnings.warn(msg(x))

    if not result:
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))
    else:
        return result


def check_cyc(lst, errors="raise"):
    """ Validate a CYC option and returned it as a list of integers

    Parameters
    ----------
    cyc: int
        CYC must be an integer or an iterable with elements that can be casted as positive integers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

    Returns
    -------
    list(int)
    """
    is_cyc(lst, errors=errors)

    # Make sure we deal with a list
    lst = to_list(lst)

    # Then cast list elements as integers
    return [abs(int(x)) for x in lst]


def is_cyc(lst, errors="raise"):  # noqa: C901
    """ Check if a CYC is valid
    Parameters
    ----------
    cyc: int, list(int), array(int)
        CYC must be a single or a list of at most 4 digit positive numbers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.
    Returns
    -------
    bool
        True if cyc is indeed a list of integers
    """
    # Make sure we deal with a list
    lst = to_list(lst)

    # Error message:
    msg = "CYC must be a single or a list of at most 4 digit positive numbers. Invalid: '{}'".format

    # Then try to cast list elements as integers, return True if ok
    result = True
    try:
        for x in lst:
            if not str(x).isdigit():
                result = False

            if (len(str(x)) > 4):
                result = False

            if int(x) < 0:
                result = False

    except Exception:
        result = False
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))

    if not result:
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))
    else:
        return result


def check_index_cols(column_names: list, convention: str = 'ar_index_global_prof'):
    """
        ar_index_global_prof.txt: Index of profile files
        Profile directory file of the Argo Global Data Assembly Center
        file,date,latitude,longitude,ocean,profiler_type,institution,date_update

        argo_bio-profile_index.txt: bgc Argo profiles index file
        The directory file describes all individual bio-profile files of the argo GDAC ftp site.
        file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
    """
    # Default for 'ar_index_global_prof'
    ref = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
           'date_update']
    if convention == 'argo_bio-profile_index' or convention == 'argo_synthetic-profile_index':
        ref = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
               'parameters', 'parameter_data_mode', 'date_update']

    if not is_list_equal(column_names, ref):
        # log.debug("Expected: %s, got: %s" % (";".join(ref), ";".join(column_names)))
        raise InvalidDatasetStructure("Unexpected column names in this index !")
    else:
        return column_names

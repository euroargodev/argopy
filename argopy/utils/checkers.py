import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
from fsspec.core import split_protocol
import fsspec
from socket import gaierror
import urllib
import json
import logging

from ..options import OPTIONS
from ..errors import InvalidDatasetStructure, FtpPathError, InvalidFetcher
from .lists import list_available_data_src, list_available_index_src
from .casting import to_list


log = logging.getLogger("argopy.utils.checkers")


def is_indexbox(box: list, errors="raise"):
    """Check if this array matches a 2d or 3d index box definition

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
    """Return true if 2 lists contain same elements"""
    return len(lst1) == len(lst2) and len(lst1) == sum(
        [1 for i, j in zip(lst1, lst2) if i == j]
    )


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


def check_cyc(lst, errors="raise"):
    """Validate a CYC option and returned it as a list of integers

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
    """Check if a CYC is valid
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

            if len(str(x)) > 4:
                result = False

            if int(x) < 0:
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


def check_index_cols(column_names: list, convention: str = "ar_index_global_prof"):
    """
    ar_index_global_prof.txt: Index of profile files
    Profile directory file of the Argo Global Data Assembly Center
    file,date,latitude,longitude,ocean,profiler_type,institution,date_update

    argo_bio-profile_index.txt: bgc Argo profiles index file
    The directory file describes all individual bio-profile files of the argo GDAC ftp site.
    file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
    """
    # Default for 'ar_index_global_prof'
    ref = [
        "file",
        "date",
        "latitude",
        "longitude",
        "ocean",
        "profiler_type",
        "institution",
        "date_update",
    ]
    if (
        convention == "argo_bio-profile_index"
        or convention == "argo_synthetic-profile_index"
    ):
        ref = [
            "file",
            "date",
            "latitude",
            "longitude",
            "ocean",
            "profiler_type",
            "institution",
            "parameters",
            "parameter_data_mode",
            "date_update",
        ]

    if not is_list_equal(column_names, ref):
        # log.debug("Expected: %s, got: %s" % (";".join(ref), ";".join(column_names)))
        raise InvalidDatasetStructure("Unexpected column names in this index !")
    else:
        return column_names


def check_gdac_path(path, errors="ignore"):  # noqa: C901
    """Check if a path has the expected GDAC ftp structure

    Expected GDAC ftp structure::

        .
        └── dac
            ├── aoml
            ├── ...
            ├── coriolis
            ├── ...
            ├── meds
            └── nmdis

    This check will return True if at least one DAC sub-folder is found under path/dac/<dac_name>

    Examples::
    >>> check_gdac_path("https://data-argo.ifremer.fr")  # True
    >>> check_gdac_path("ftp://ftp.ifremer.fr/ifremer/argo") # True
    >>> check_gdac_path("ftp://usgodae.org/pub/outgoing/argo") # True
    >>> check_gdac_path("/home/ref-argo/gdac") # True
    >>> check_gdac_path("https://www.ifremer.fr") # False
    >>> check_gdac_path("ftp://usgodae.org/pub/outgoing") # False

    Parameters
    ----------
    path: str
        Path name to check, including access protocol
    errors: str
        "ignore" or "raise" (or "warn")

    Returns
    -------
    checked: boolean
        True if at least one DAC folder is found under path/dac/<dac_name>
        False otherwise
    """
    # Create a file system for this path
    if split_protocol(path)[0] is None:
        fs = fsspec.filesystem("file")
    elif "https" in split_protocol(path)[0]:
        fs = fsspec.filesystem("http")
    elif "ftp" in split_protocol(path)[0]:
        try:
            host = split_protocol(path)[-1].split("/")[0]
            fs = fsspec.filesystem("ftp", host=host)
        except gaierror:
            if errors == "raise":
                raise FtpPathError("Can't get address info (GAIerror) on '%s'" % host)
            elif errors == "warn":
                warnings.warn("Can't get address info (GAIerror) on '%s'" % host)
                return False
            else:
                return False
    else:
        raise FtpPathError(
            "Unknown protocol for an Argo GDAC host: %s" % split_protocol(path)[0]
        )

    # dacs = [
    #     "aoml",
    #     "bodc",
    #     "coriolis",
    #     "csio",
    #     "csiro",
    #     "incois",
    #     "jma",
    #     "kma",
    #     "kordi",
    #     "meds",
    #     "nmdis",
    # ]

    # Case 1:
    check1 = (
        fs.exists(path)
        and fs.exists(fs.sep.join([path, "dac"]))
        # and np.any([fs.exists(fs.sep.join([path, "dac", dac])) for dac in dacs])  # Take too much time on http/ftp GDAC server
    )
    if check1:
        return True
    elif errors == "raise":
        raise FtpPathError(
            "This path is not GDAC compliant (no `dac` folder with legitimate sub-folder):\n%s"
            % path
        )

    elif errors == "warn":
        warnings.warn("This path is not GDAC compliant:\n%s" % path)
        return False
    else:
        return False


def isconnected(host: str = "https://www.ifremer.fr", maxtry: int = 10):
    """Check if an URL is alive

    Parameters
    ----------
    host: str
        URL to use, 'https://www.ifremer.fr' by default
    maxtry: int, default: 10
        Maximum number of host connections to try before

    Returns
    -------
    bool
    """
    # log.debug("isconnected: %s" % host)
    if split_protocol(host)[0] in ["http", "https", "ftp", "sftp"]:
        it = 0
        while it < maxtry:
            try:
                # log.debug("Checking if %s is connected ..." % host)
                urllib.request.urlopen(
                    host, timeout=1
                )  # nosec B310 because host protocol already checked
                result, it = True, maxtry
            except Exception:
                result, it = False, it + 1
        return result
    else:
        return os.path.exists(host)


def urlhaskeyword(url: str = "", keyword: str = "", maxtry: int = 10):
    """Check if a keyword is in the content of a URL

    Parameters
    ----------
    url: str
    keyword: str
    maxtry: int, default: 10
        Maximum number of host connections to try before returning False

    Returns
    -------
    bool
    """
    it = 0
    while it < maxtry:
        try:
            with fsspec.open(url) as f:
                data = f.read()
            result = keyword in str(data)
            it = maxtry
        except Exception:
            result, it = False, it + 1
    return result


def isalive(api_server_check: Union[str, dict] = "") -> bool:
    """Check if an API is alive or not

    2 methods are available:

    - URL Ping
    - keyword Check

    Parameters
    ----------
    api_server_check
        Url string or dictionary with [``url``, ``keyword``] keys.

        - For a string, uses: :class:`argopy.utilities.isconnected`
        - For a dictionary,  uses: :class:`argopy.utilities.urlhaskeyword`

    Returns
    -------
    bool
    """
    # log.debug("isalive: %s" % api_server_check)
    if isinstance(api_server_check, dict):
        return urlhaskeyword(
            url=api_server_check["url"], keyword=api_server_check["keyword"]
        )
    else:
        return isconnected(api_server_check)


def isAPIconnected(src="erddap", data=True):
    """Check if a source API is alive or not

    The API is connected when it has a live URL or valid folder path.

    Parameters
    ----------
    src: str
        The data or index source name, 'erddap' default
    data: bool
        If True check the data fetcher (default), if False, check the index fetcher

    Returns
    -------
    bool
    """
    if data:
        list_src = list_available_data_src()
    else:
        list_src = list_available_index_src()

    if src in list_src and getattr(list_src[src], "api_server_check", None):
        return isalive(list_src[src].api_server_check)
    else:
        raise InvalidFetcher


def erddap_ds_exists(
    ds: Union[list, str] = "ArgoFloats", erddap: str = None, maxtry: int = 2
) -> bool:
    """Check if a dataset exists on a remote erddap server

    Parameter
    ---------
    ds: str, default='ArgoFloats'
        Name of the erddap dataset to check
    erddap: str, default=OPTIONS['erddap']
        Url of the erddap server
    maxtry: int, default: 2
        Maximum number of host connections to try

    Return
    ------
    bool
    """
    if erddap is None:
        erddap = OPTIONS["erddap"]
    # log.debug("from erddap_ds_exists: %s" % erddap)
    if isconnected(erddap, maxtry=maxtry):
        from ..stores import httpstore  # must import here to avoid circular import

        with httpstore(timeout=OPTIONS["api_timeout"]).open(
            "".join([erddap, "/info/index.json"])
        ) as of:
            erddap_index = json.load(of)
        if is_list_of_strings(ds):
            return [
                this_ds in [row[-1] for row in erddap_index["table"]["rows"]]
                for this_ds in ds
            ]
        else:
            return ds in [row[-1] for row in erddap_index["table"]["rows"]]
    else:
        log.debug("Cannot reach erddap server: %s" % erddap)
        warnings.warn(
            "Return False because we cannot reach the erddap server %s" % erddap
        )
        return False

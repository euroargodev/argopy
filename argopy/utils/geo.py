import numpy as np
import pandas as pd

from ..options import OPTIONS


def wrap_longitude(grid_long):
    """Allows longitude (0-360) to wrap beyond the 360 mark, for mapping purposes.

    Makes sure that, if the longitude is near the boundary (0 or 360) that we
    wrap the values beyond 360, so it appears nicely on a map
    This is a refactor between get_region_data and get_region_hist_locations to
    avoid duplicate code

    source:
    https://github.com/euroargodev/argodmqc_owc/blob/e174f4538fdae1534c9740491398972b1ffec3ca/pyowc/utilities.py#L80

    Parameters
    ----------
    grid_long: array of longitude values

    Returns
    -------
    array of longitude values that can extend past 360
    """
    neg_long = np.argwhere(grid_long < 0)
    grid_long[neg_long] = grid_long[neg_long] + 360

    # if we have data close to upper boundary (360), then wrap some of the data round
    # so it appears on the map
    top_long = np.argwhere(grid_long >= 320)
    if top_long.__len__() != 0:
        bottom_long = np.argwhere(grid_long <= 40)
        grid_long[bottom_long] = 360 + grid_long[bottom_long]

    return grid_long


def conv_lon(x, conv: str = "180"):
    """Apply longitude convention to array x

    Params
    ------
    x:
        Object to iterate over (e.g. list, numpy array,...)
    conv: str, default='180'
        Convention to apply, must be '180' or '360'
    Returns
    -------
    Transformed x
    """
    if conv == "360":
        c = lambda x: type(x)(  # noqa: E731
            np.where(np.logical_and(x >= -180.0, x < 0.0), x + 360.0, x)[np.newaxis][0]
        )
    elif conv == "180":
        c = lambda x: type(x)(  # noqa: E731
            np.where(x > 180.0, x - 360.0, x)[np.newaxis][0]
        )
    else:
        return x

    return np.frompyfunc(c, 1, 1)(x)


def wmo2box(wmo_id: int):
    """Convert WMO square box number into a latitude/longitude box

    See:
    https://en.wikipedia.org/wiki/World_Meteorological_Organization_squares
    https://commons.wikimedia.org/wiki/File:WMO-squares-global.gif

    Parameters
    ----------
    wmo_id: int
        WMO square number, must be between 1000 and 7817

    Returns
    -------
    box: list(int)
        [lon_min, lon_max, lat_min, lat_max] bounds to the WMO square number

    Notes
    -----
    Longitude values are returned according the argopy option "longitude_convention", which is '180' or '360'.
    """
    if wmo_id < 1000 or wmo_id > 7817:
        raise ValueError("Invalid WMO square number, must be between 1000 and 7817.")
    wmo_id = str(wmo_id)

    # "global quadrant" numbers where 1=NE, 3=SE, 5=SW, 7=NW
    quadrant = int(wmo_id[0])
    if quadrant not in [1, 3, 5, 7]:
        raise ValueError("Invalid WMO square number, 1st digit must be 1, 3, 5 or 7.")

    # 'minimum' Latitude square boundary, nearest to the Equator
    nearest_to_the_Equator_latitude = int(wmo_id[1])

    # 'minimum' Longitude square boundary, nearest to the Prime Meridian
    nearest_to_the_Prime_Meridian = int(wmo_id[2:4])

    #
    dd = 10
    if quadrant in [1, 3]:
        lon_min = nearest_to_the_Prime_Meridian * dd
        lon_max = nearest_to_the_Prime_Meridian * dd + dd
    elif quadrant in [5, 7]:
        lon_min = -nearest_to_the_Prime_Meridian * dd - dd
        lon_max = -nearest_to_the_Prime_Meridian * dd

    if quadrant in [1, 7]:
        lat_min = nearest_to_the_Equator_latitude * dd
        lat_max = nearest_to_the_Equator_latitude * dd + dd
    elif quadrant in [3, 5]:
        lat_min = -nearest_to_the_Equator_latitude * dd - dd
        lat_max = -nearest_to_the_Equator_latitude * dd

    box = [
        conv_lon(lon_min, OPTIONS["longitude_convention"]),
        conv_lon(lon_max, OPTIONS["longitude_convention"]),
        lat_min,
        lat_max,
    ]
    return box


def toYearFraction(
    this_date: pd._libs.tslibs.timestamps.Timestamp = pd.to_datetime("now", utc=True)
):
    """Compute decimal year, robust to leap years, precision to the second

    Compute the fraction of the year a given timestamp corresponds to.
    The "fraction of the year" goes:
    - from 0 on 01-01T00:00:00.000 of the year
    - to 1 on the 01-01T00:00:00.000 of the following year

    1 second corresponds to the number of days in the year times 86400.
    The faction of the year is rounded to 10-digits in order to have a "second" precision.

    See discussion here: https://github.com/euroargodev/argodmqc_owc/issues/35

    Parameters
    ----------
    pd._libs.tslibs.timestamps.Timestamp

    Returns
    -------
    float
    """
    if "UTC" in [this_date.tzname() if this_date.tzinfo is not None else ""]:
        startOfThisYear = pd.to_datetime(
            "%i-01-01T00:00:00.000" % this_date.year, utc=True
        )
    else:
        startOfThisYear = pd.to_datetime("%i-01-01T00:00:00.000" % this_date.year)
    yearDuration_sec = (
        startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear
    ).total_seconds()

    yearElapsed_sec = (this_date - startOfThisYear).total_seconds()
    fraction = yearElapsed_sec / yearDuration_sec
    fraction = np.round(fraction, 10)
    return this_date.year + fraction


def YearFraction_to_datetime(yf: float):
    """Compute datetime from year fraction

    Inverse the toYearFraction() function

    Parameters
    ----------
    float

    Returns
    -------
    pd._libs.tslibs.timestamps.Timestamp
    """
    year = np.int32(yf)
    fraction = yf - year
    fraction = np.round(fraction, 10)

    startOfThisYear = pd.to_datetime("%i-01-01T00:00:00" % year)
    yearDuration_sec = (
        startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear
    ).total_seconds()
    yearElapsed_sec = pd.Timedelta(fraction * yearDuration_sec, unit="s")
    return pd.to_datetime(startOfThisYear + yearElapsed_sec, unit="s")

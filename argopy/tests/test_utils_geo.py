import pytest
import numpy as np
import pandas as pd
from argopy.utils.geo import wmo2box, wrap_longitude, conv_lon, toYearFraction, YearFraction_to_datetime
from argopy.utils.geo import split_box_at_dateline, lon_range_to_180
from argopy.utils.checkers import is_box


def test_wmo2box():
    with pytest.raises(ValueError):
        wmo2box(12)
    with pytest.raises(ValueError):
        wmo2box(8000)
    with pytest.raises(ValueError):
        wmo2box(2000)

    def complete_box(b):
        b2 = b.copy()
        b2.insert(4, 0.)
        b2.insert(5, 10000.)
        return b2

    assert is_box(complete_box(wmo2box(1212)))
    assert is_box(complete_box(wmo2box(3324)))
    assert is_box(complete_box(wmo2box(5402)))
    assert is_box(complete_box(wmo2box(7501)))


def test_wrap_longitude():
    assert wrap_longitude(np.array([-20])) == 340
    assert wrap_longitude(np.array([40])) == 40
    assert np.all(np.equal(wrap_longitude(np.array([340, 20])), np.array([340, 380])))


def test_conv_lon():
    assert conv_lon(-5, conv='180') == -5
    assert conv_lon(-5, conv='360') == 355
    assert conv_lon(355, conv='180') == -5
    assert conv_lon(355, conv='360') == 355
    assert conv_lon(12, conv='toto') == 12


def test_split_box_at_dateline():
    # Going through the date line: split in two boxes, other box limits unchanged
    assert split_box_at_dateline([175, 185, -5, 5, 0, 100]) == [[175, 180, -5, 5, 0, 100], [180, 185, -5, 5, 0, 100]]
    assert split_box_at_dateline([0, 360, -5, 5]) == [[0, 180, -5, 5], [180, 360, -5, 5]]
    # Not going through the date line: nothing to split
    assert split_box_at_dateline([185, 190, -5, 5]) == [[185, 190, -5, 5]]
    assert split_box_at_dateline([175, 180, -5, 5]) == [[175, 180, -5, 5]]
    assert split_box_at_dateline([180, 185, -5, 5]) == [[180, 185, -5, 5]]
    assert split_box_at_dateline([-20, -16, -5, 5]) == [[-20, -16, -5, 5]]  # '180' convention


def test_lon_range_to_180():
    # East of the date line in the '360' convention: shifted to negative longitudes
    assert lon_range_to_180(180, 185) == (-180, -175)
    assert lon_range_to_180(185, 190) == (-175, -170)
    assert lon_range_to_180(180, 360) == (-180, 0)
    # Already in [-180, 180]: unchanged
    assert lon_range_to_180(175, 180) == (175, 180)
    assert lon_range_to_180(-20, -16) == (-20, -16)


def test_toYearFraction():
    assert toYearFraction(pd.to_datetime('202001010000')) == 2020
    assert toYearFraction(pd.to_datetime('202001010000', utc=True)) == 2020
    assert toYearFraction(pd.to_datetime('202001010000')+pd.offsets.DateOffset(years=1)) == 2021


def test_YearFraction_to_datetime():
    assert YearFraction_to_datetime(2020) == pd.to_datetime('202001010000')
    assert YearFraction_to_datetime(2020+1) == pd.to_datetime('202101010000')

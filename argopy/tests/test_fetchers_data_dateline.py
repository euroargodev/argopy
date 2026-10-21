"""
Test data fetchers with a box going through the date line (180 degrees longitude)

Argo data are stored with longitudes in [-180, 180]. With the '360' longitude convention, a user can ask for a box
like [175, 185]: fetchers must then select data from 175 to 180 and from -180 to -175.

These tests don't need any server. The argovis fetcher is not tested here, its API handles longitudes above 180.
"""

import re
from types import SimpleNamespace
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from aiohttp import ClientResponseError

import argopy
from argopy.errors import ErddapServerError
from argopy.data_fetchers.erddap_data import Fetch_box as Erddap_Fetch_box
from argopy.data_fetchers.gdac_data_processors import filter_points
from argopy.tests.helpers.utils import requires_ipython


# Latitude, pressure and time limits of all the boxes tested here:
OTHER_LIMITS = [-5, 5, 0, 100.0, "2026-06-01", "2026-09-01"]


########
# ERDDAP: which longitudes are requested ?
########

SERVER = "http://127.0.0.1:1/erddap"  # Never reached, we only build request URLs


def requested_longitudes(uris):
    """Longitude range of each request URL, e.g. [(175.0, 180.0), (-180.0, -175.0)]"""
    ranges = []
    for uri in uris:
        url = unquote(uri)
        lon_min = float(re.search(r"longitude>=(-?[\d.]+)", url).group(1))
        lon_max = float(re.search(r"longitude<=(-?[\d.]+)", url).group(1))
        ranges.append((lon_min, lon_max))
    return ranges


# User box (with the '360' convention) and the longitude ranges to request (in the [-180, 180] convention of Argo data)
BOXES = [
    pytest.param([175, 185], [(175, 180), (-180, -175)], id="through the date line"),  # 2 requests
    pytest.param([185, 190], [(-175, -170)], id="east of the date line"),  # shifted to negative longitudes
    pytest.param([170, 175], [(170, 175)], id="west of the date line"),  # nothing to change
]


@pytest.mark.parametrize("user_box, requests", BOXES)
def test_erddap_requests(user_box, requests):
    with argopy.set_options(longitude_convention="360"):
        fetcher = Erddap_Fetch_box(box=user_box + OTHER_LIMITS, ds="phy", server=SERVER)
        assert requested_longitudes(fetcher.uri) == requests


@pytest.mark.parametrize("user_box, requests", BOXES)
def test_erddap_requests_parallel(user_box, requests):
    """With the parallel option the box is cut in more requests, but together they must cover the same longitudes"""
    with argopy.set_options(longitude_convention="360"):
        fetcher = Erddap_Fetch_box(
            box=user_box + OTHER_LIMITS,
            ds="phy",
            server=SERVER,
            parallel=True,
            # Cut the box in small pieces. With 4 degrees, [175, 185] gives a piece from 178.3 to 181.7, across the date line:
            chunks_maxsize={"lon": 4, "lat": 5, "dpt": 100},
        )
        pieces = requested_longitudes(fetcher.uri)

    for lon_min, lon_max in requests:
        pieces_here = [(a, b) for a, b in pieces if lon_min <= a and b <= lon_max]
        assert min(a for a, _ in pieces_here) == lon_min
        assert max(b for _, b in pieces_here) == lon_max
    assert all(any(lon_min <= a and b <= lon_max for lon_min, lon_max in requests) for a, b in pieces)


########
# ERDDAP: what is returned when the 2 requests of a box give data, no data or an error ?
########

WEST = [175.5, 179.9]  # Floats answered by the request for longitudes 175 to 180
EAST = [-179.9, -175.5]  # Floats answered by the request for longitudes -180 to -175
NO_DATA = None
SERVER_ERROR = "server error"


def load_box_through_date_line(monkeypatch, west, east):
    """Load the box [175, 185] with a fake server: ``west`` and ``east`` are its answers to the 2 requests

    An answer is a list of float longitudes, NO_DATA or SERVER_ERROR.
    """
    answers = {"west": west, "east": east}

    def fake_open_dataset(url, *args, **kwargs):
        side = "east" if "longitude>=-180" in unquote(url) else "west"
        answer = answers[side]
        if answer is NO_DATA:
            raise FileNotFoundError(url)  # This is how a request with no data is reported
        if answer is SERVER_ERROR:
            raise ClientResponseError(SimpleNamespace(real_url=url), (), status=502, message="Proxy Error")
        return xr.Dataset({"LONGITUDE": ("N_POINTS", answer)}, coords={"N_POINTS": np.arange(len(answer))})

    # Our fake floats are not complete erddap data, so skip the pre-processing:
    monkeypatch.setattr("argopy.data_fetchers.erddap_data.pre_process", lambda ds, **kwargs: ds)

    with argopy.set_options(longitude_convention="360"):
        fetcher = Erddap_Fetch_box(box=[175, 185] + OTHER_LIMITS, ds="phy", server=SERVER)
        monkeypatch.setattr(fetcher.fs, "open_dataset", fake_open_dataset)
        return fetcher.to_xarray()


def test_erddap_data_on_both_sides(monkeypatch):
    ds = load_box_through_date_line(monkeypatch, west=WEST, east=EAST)
    assert sorted(ds["LONGITUDE"].values) == sorted(WEST + EAST)
    assert list(ds["N_POINTS"].values) == [0, 1, 2, 3]  # Points of the 2 requests are numbered as one dataset


@pytest.mark.parametrize(
    "west, east, expected",
    [(NO_DATA, EAST, EAST), (WEST, NO_DATA, WEST)],
    ids=["no data in the west", "no data in the east"],
)
def test_erddap_no_data_on_one_side(monkeypatch, west, east, expected):
    """No data on one side of the date line is fine: we get the data of the other side"""
    ds = load_box_through_date_line(monkeypatch, west=west, east=east)
    assert sorted(ds["LONGITUDE"].values) == sorted(expected)


@requires_ipython  # Failed requests are only reported by the thread pool executor when IPython is installed
@pytest.mark.parametrize(
    "west, east",
    [(SERVER_ERROR, EAST), (WEST, SERVER_ERROR)],
    ids=["error in the west", "error in the east"],
)
def test_erddap_server_error_on_one_side(monkeypatch, west, east):
    """A server error on one side must raise, otherwise we would silently return half of the box"""
    with pytest.raises(ErddapServerError):
        load_box_through_date_line(monkeypatch, west=west, east=east)


########
# GDAC: which downloaded points are kept ?
########


@pytest.mark.parametrize(
    "user_box, points, kept",
    [
        (
            [175, 185],
            [170, 175.5, 179.9, 180, -180, -179.9, -175.5, -170],
            [175.5, 179.9, 180, -180, -179.9, -175.5],  # 180 and -180 are the same place
        ),
        ([185, 190], [-179, -174.5, -170.5, -165], [-174.5, -170.5]),
        ([170, 175], [170.5, 174.5, 179, -179], [170.5, 174.5]),
    ],
    ids=["through the date line", "east of the date line", "west of the date line"],
)
def test_gdac_filter(user_box, points, kept):
    """One point per longitude, all inside the latitude, pressure and time limits of the box"""
    n = len(points)
    ds = xr.Dataset(
        {
            "LONGITUDE": ("N_POINTS", np.array(points)),
            "LATITUDE": ("N_POINTS", np.zeros(n)),
            "PRES": ("N_POINTS", np.full(n, 50.0)),
            "TIME": ("N_POINTS", np.full(n, pd.Timestamp("2026-07-15").to_datetime64())),
        },
        coords={"N_POINTS": np.arange(n)},
    )
    with argopy.set_options(longitude_convention="360"):
        ds = filter_points(ds, access_point="BOX", BOX=user_box + OTHER_LIMITS)

    assert list(ds["LONGITUDE"].values) == kept

"""
Test data fetchers with a box going through the date line (180 degrees longitude)

Argo data are stored with longitudes in [-180, 180]. With the '360' longitude convention, a user can define a box
going through the date line, e.g. [175, 185]: data fetchers must then select data from 175 to 180 and from -180 to -175.

These tests don't need any server:
- for the erddap fetcher, we check the longitude ranges written in the request URLs,
- for the gdac fetcher, we check the filter applied on downloaded points, with a few sample floats.

The argovis fetcher is not tested here, because the Argovis API handles longitudes larger than 180 itself.
"""

import re
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import argopy
from argopy.data_fetchers.erddap_data import Fetch_box as Erddap_Fetch_box
from argopy.data_fetchers.gdac_data_processors import filter_points


"""
List boxes to be tested

For each box, we give:
- 'lon_360': the longitude range requested by the user, in the '360' convention,
- 'expected': the same longitude range(s) in the [-180, 180] convention of Argo data,
- 'inside': positions of sample floats inside the box, they must be selected,
- 'outside': positions of sample floats outside the box, they must not be selected.
"""
BOXES = [
    {
        "name": "going through the date line",
        "lon_360": [175, 185],
        "expected": [(175, 180), (-180, -175)],
        "inside": [175.5, 179.9, 180.0, -180.0, -179.9, -175.5],  # Points on the date line can be at 180 or -180
        "outside": [170.0, -170.0],
    },
    {
        "name": "east of the date line",
        "lon_360": [185, 190],
        "expected": [(-175, -170)],
        "inside": [-174.5, -170.5],
        "outside": [-179.0, -165.0],
    },
    {
        "name": "west of the date line (no conversion needed)",
        "lon_360": [170, 175],
        "expected": [(170, 175)],
        "inside": [170.5, 174.5],
        "outside": [179.0, -179.0],
    },
]
BOXES_IDS = ["%s %s" % (b["name"], b["lon_360"]) for b in BOXES]

"""All other box limits: latitude, pressure and time"""
LAT_PRES_TIM = [-5, 5, 0, 100.0, "2026-06-01", "2026-09-01"]


def positions(longitudes, none="none"):
    """Format a list of longitudes, e.g. '-179.9, 175.5'"""
    return ", ".join(["%g" % lon for lon in sorted(longitudes)]) or none


def ranges(lon_ranges):
    """Format a list of longitude ranges, e.g. '175 to 180, -180 to -175'"""
    return ", ".join(["%g to %g" % r for r in lon_ranges])


class Test_Erddap_DatelineBox:
    """Does the erddap fetcher request data from the right longitudes ?

    We only build the request URLs, the server is never reached.
    """

    server = "http://127.0.0.1:1/erddap"

    @staticmethod
    def requested_lon_ranges(uris):
        """Read the longitude range of each erddap request URL, e.g. 'longitude>=175&longitude<=185' gives (175, 185)"""
        lon_ranges = []
        for uri in uris:
            query = unquote(uri)
            lon_min = float(re.search(r"longitude>=(-?[\d.]+)", query).group(1))
            lon_max = float(re.search(r"longitude<=(-?[\d.]+)", query).group(1))
            if (lon_min, lon_max) not in lon_ranges:  # Chunked requests can have the same longitude range
                lon_ranges.append((lon_min, lon_max))
        return lon_ranges

    def assert_requests(self, box, uris):
        requested = self.requested_lon_ranges(uris)

        def is_requested(lon):
            return any([lon_min <= lon <= lon_max for lon_min, lon_max in requested])

        missed = [lon for lon in box["inside"] if not is_requested(lon)]
        returned_outside = [lon for lon in box["outside"] if is_requested(lon)]

        request_is_correct = not missed and not returned_outside
        assert request_is_correct, (
            "\n\nWith longitude_convention='360', the box %s is not requested correctly.\n"
            "  Expected requests covering longitudes : %s\n"
            "  Got requests for longitudes           : %s\n"
            "  Sample floats missed at positions     : %s\n"
            "  Sample floats outside the box returned: %s\n"
            % (
                box["lon_360"],
                ranges(box["expected"]),
                ranges(requested),
                positions(missed),
                positions(returned_outside, none="none, good"),
            )
        )

    @pytest.mark.parametrize("box", BOXES, indirect=False, ids=BOXES_IDS)
    def test_request(self, box):
        with argopy.set_options(longitude_convention="360"):
            fetcher = Erddap_Fetch_box(box=box["lon_360"] + LAT_PRES_TIM, ds="phy", server=self.server)
            self.assert_requests(box, fetcher.uri)

    @pytest.mark.parametrize("box", BOXES, indirect=False, ids=BOXES_IDS)
    def test_request_parallel(self, box):
        with argopy.set_options(longitude_convention="360"):
            fetcher = Erddap_Fetch_box(
                box=box["lon_360"] + LAT_PRES_TIM,
                ds="phy",
                server=self.server,
                parallel=True,
                chunks_maxsize={"lon": 2.5, "lat": 5, "dpt": 100},  # Make sure the box is split in several requests
            )
            self.assert_requests(box, fetcher.uri)


class Test_Gdac_DatelineBox:
    """Does the gdac fetcher keep the right points, once data are downloaded ?

    We apply the gdac points filter to a few sample floats, one point at each position to test.
    """

    @staticmethod
    def sample_floats_at(longitudes):
        """Create one point at each longitude, all inside the box latitude, pressure and time limits"""
        n = len(longitudes)
        return xr.Dataset(
            {
                "LONGITUDE": ("N_POINTS", np.array(longitudes)),
                "LATITUDE": ("N_POINTS", np.zeros(n)),
                "PRES": ("N_POINTS", np.full(n, 50.0)),
                "TIME": ("N_POINTS", np.full(n, pd.Timestamp("2026-07-15").to_datetime64())),
            },
            coords={"N_POINTS": np.arange(n)},
        )

    @pytest.mark.parametrize("box", BOXES, indirect=False, ids=BOXES_IDS)
    def test_filter(self, box):
        ds = self.sample_floats_at(box["inside"] + box["outside"])
        with argopy.set_options(longitude_convention="360"):
            ds = filter_points(ds, access_point="BOX", BOX=box["lon_360"] + LAT_PRES_TIM)

        kept = list(ds["LONGITUDE"].values)
        missed = [lon for lon in box["inside"] if lon not in kept]
        kept_outside = [lon for lon in box["outside"] if lon in kept]

        filter_is_correct = not missed and not kept_outside
        assert filter_is_correct, (
            "\n\nWith longitude_convention='360', the box %s is not filtered correctly.\n"
            "  Expected to keep longitudes        : %s\n"
            "  Sample floats given at positions   : %s\n"
            "  Sample floats kept at positions    : %s\n"
            "  Sample floats missed at positions  : %s\n"
            "  Sample floats outside the box kept : %s\n"
            % (
                box["lon_360"],
                ranges(box["expected"]),
                positions(box["inside"] + box["outside"]),
                positions(kept),
                positions(missed),
                positions(kept_outside, none="none, good"),
            )
        )

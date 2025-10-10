import pytest
import logging
import numpy as np

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver


log = logging.getLogger("argopy.tests.extensions.canyonb")
USE_MOCKED_SERVER = True


@pytest.fixture
def fetcher():
    defaults_args = {
        "src": "erddap",
        "cache": False,
        "ds": "bgc",
        "params": "DOXY",
        "measured": "DOXY",
    }
    if USE_MOCKED_SERVER:
        defaults_args["server"] = mocked_server_address

    return DataFetcher(**defaults_args).profile(5903248, 34)


@pytest.mark.parametrize(
    "what",
    [None, "PO4", ["PO4", "pHT"]],
    indirect=False,
)
def test_predict(fetcher, what, mocked_erddapserver):
    """Test CANYON-B predictions for various parameters"""
    ds = fetcher.to_xarray()
    ds = ds.argo.canyon_b.predict(what)
    assert "CANYON-B" in ds.attrs["Processing_history"]


@pytest.mark.parametrize(
    "param",
    ["PO4", "NO3", "SiOH4", "AT", "DIC", "pHT", "pCO2"],
    indirect=False,
)
def test_predict_single_param(fetcher, param, mocked_erddapserver):
    """Test CANYON-B prediction for each parameter individually"""
    ds = fetcher.to_xarray()
    ds = ds.argo.canyon_b.predict(param)

    assert param in ds
    assert f"{param}_CI" in ds
    assert f"{param}_CIM" in ds
    assert f"{param}_CIN" in ds
    assert f"{param}_CII" in ds


def test_predict_with_custom_errors(fetcher, mocked_erddapserver):
    """Test CANYON-B prediction with custom input errors"""
    ds = fetcher.to_xarray()
    ds = ds.argo.canyon_b.predict(
        "PO4", epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01
    )

    assert "PO4" in ds
    assert "PO4_CI" in ds
    assert "PO4_CIM" in ds
    assert "PO4_CIN" in ds
    assert "PO4_CII" in ds


def test_predict_invalid_param(fetcher, mocked_erddapserver):
    """Test that invalid parameter raises ValueError"""
    ds = fetcher.to_xarray()

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds.argo.canyon_b.predict("INVALID_PARAM")


def test_ds2df(fetcher, mocked_erddapserver):
    """Test conversion from dataset to dataframe"""
    ds = fetcher.to_xarray()
    df = ds.argo.canyon_b.ds2df()

    required_cols = ["lat", "lon", "dec_year", "temp", "psal", "doxy", "pres"]
    for col in required_cols:
        assert col in df.columns

    assert not np.array_equal(df["pres"].values, ds["PRES"].values)


def test_create_canyonb_input_matrix(fetcher, mocked_erddapserver):
    """Test creation of CANYON-B input matrix"""
    ds = fetcher.to_xarray()
    data = ds.argo.canyon_b.create_canyonb_input_matrix()

    assert data.shape[1] == 8


def test_adjust_arctic_latitude(fetcher, mocked_erddapserver):
    """Test Arctic latitude adjustment"""
    ds = fetcher.to_xarray()

    lat = np.array([70.0, 80.0, 85.0])
    lon = np.array([-100.0, 150.0, 0.0])
    adjusted_lat = ds.argo.canyon_b.adjust_arctic_latitude(lat, lon)
    assert adjusted_lat.shape == lat.shape

    lat_non_arctic = np.array([30.0, 40.0, 50.0])
    lon = np.array([0.0, 10.0, 20.0])
    adjusted_lat_non_arctic = ds.argo.canyon_b.adjust_arctic_latitude(
        lat_non_arctic, lon
    )
    assert np.allclose(adjusted_lat_non_arctic, lat_non_arctic)


def test_load_weights(fetcher, mocked_erddapserver):
    """Test loading of CANYON-B weights"""
    ds = fetcher.to_xarray()

    for param in ["AT", "DIC", "pCO2", "NO3", "PO4", "SiOH4", "pHT"]:
        weights = ds.argo.canyon_b.load_weights(param)
        assert hasattr(weights, "shape")
        assert weights.shape[0] > 0
        assert weights.shape[1] > 0
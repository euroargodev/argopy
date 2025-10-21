import pytest
import logging
import numpy as np

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver
from utils import requires_pyco2sys


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


@requires_pyco2sys
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


@requires_pyco2sys
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


@requires_pyco2sys
def test_predict_with_custom_errors(fetcher, mocked_erddapserver):
    """Test CANYON-B prediction with custom input errors"""
    ds = fetcher.to_xarray()
    ds = ds.argo.canyon_b.predict(
        "PO4", epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01
    )

    assert "PO4" in ds


@requires_pyco2sys
def test_predict_with_array_edoxy(fetcher, mocked_erddapserver):
    """Test CANYON-B prediction with array oxygen errors"""
    ds = fetcher.to_xarray()
    nol = ds.argo.N_POINTS
    edoxy_array = np.full(nol, 0.01)

    ds = ds.argo.canyon_b.predict('PO4', edoxy=edoxy_array)
    assert 'PO4' in ds


@requires_pyco2sys
def test_predict_invalid_param(fetcher, mocked_erddapserver):
    """Test that invalid parameter raises ValueError"""
    ds = fetcher.to_xarray()

    with pytest.raises(ValueError, match="Invalid parameter"):
        ds.argo.canyon_b.predict("INVALID_PARAM")


@requires_pyco2sys
def test_predict_private_uncertainties(fetcher, mocked_erddapserver):
    """Test that _predict() returns all uncertainty components"""
    ds = fetcher.to_xarray()
    result = ds.argo.canyon_b._predict('PO4')

    # Check that all uncertainty components are present
    assert 'PO4' in result
    assert 'PO4_ci' in result
    assert 'PO4_cim' in result
    assert 'PO4_cin' in result
    assert 'PO4_cii' in result
    assert 'PO4_inx' in result

    # Check shapes are correct
    nol = ds.argo.N_POINTS
    assert result['PO4'].shape[0] == nol


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


def test_decimal_year(fetcher, mocked_erddapserver):
    """Test decimal year calculation"""
    ds = fetcher.to_xarray()
    dec_year = ds.argo.canyon_b.decimal_year
    
    # Check it has same length as data
    assert len(dec_year) == len(ds.argo.canyon_b._obj[ds.argo.canyon_b._argo._TNAME])


@pytest.mark.parametrize(
      "param,expected_unit",
      [
          ("PO4", "micromole/kg"),
          ("NO3", "micromole/kg"),
          ("pHT", "insitu total scale"),
          ("pCO2", "micro atm"),
      ],
  )

def test_get_param_attrs(fetcher, param, expected_unit, mocked_erddapserver):
    """Test parameter attributes are correctly defined"""
    ds = fetcher.to_xarray()
    attrs = ds.argo.canyon_b.get_param_attrs(param)

    assert 'units' in attrs
    assert 'long_name' in attrs
    assert 'comment' in attrs
    assert 'reference' in attrs
    assert attrs['units'] == expected_unit

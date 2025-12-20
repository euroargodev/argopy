import pytest
import logging
import numpy as np
import pandas as pd
import xarray as xr

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver
from utils import requires_pyco2sys, requires_numba

pytestmark = [requires_pyco2sys, requires_numba]

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


@pytest.fixture
def ds(fetcher, mocked_erddapserver):
    """Base dataset"""
    return fetcher.to_xarray()


@pytest.mark.parametrize(
    "what",
    [None, "PO4"],
    indirect=False,
)
def test_predict(ds, what, mocked_erddapserver):
    """Test CANYON-B predictions for various parameters"""
    ds = ds.argo.canyon_b.predict(what)

    assert "CANYON-B" in ds.attrs["Processing_history"]

    if what is not None:
        if isinstance(what, list):
            for param in what:
                assert param in ds
        else:
            assert what in ds
    else:
        # Check that all parameters are predicted
        for param in ["PO4", "NO3", "SiOH4", "AT", "DIC", "pHT", "pCO2"]:
            assert param in ds


@pytest.mark.parametrize(
    "edoxy_type",
    ["scalar", "array"],
)
def test_predict_with_custom_errors(ds, edoxy_type, mocked_erddapserver):
    """Test CANYON-B prediction with custom input errors (scalar and array)"""
    if edoxy_type == "scalar":
        edoxy = 0.01
    else:  # array
        nol = ds.argo.N_POINTS
        edoxy = np.full(nol, 0.01)

    ds = ds.argo.canyon_b.predict(
        "PO4", epres=0.5, etemp=0.005, epsal=0.005, edoxy=edoxy
    )

    assert "PO4" in ds


def test_predict_invalid_param(ds, mocked_erddapserver):
    """Test that invalid parameter raises ValueError"""
    with pytest.raises(ValueError, match="Invalid parameter"):
        ds.argo.canyon_b.predict("INVALID_PARAM")


def test_predict_private_uncertainties(ds, mocked_erddapserver):
    """Test that _predict() returns all uncertainty components"""
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


def test_ds2df(ds, mocked_erddapserver):
    """Test conversion from dataset to dataframe"""
    df = ds.argo.canyon_b.ds2df()

    required_cols = ["lat", "lon", "dec_year", "temp", "psal", "doxy", "pres"]
    for col in required_cols:
        assert col in df.columns

    assert not np.array_equal(df["pres"].values, ds["PRES"].values)


def test_create_canyonb_input_matrix(ds, mocked_erddapserver):
    """Test creation of CANYON-B input matrix"""
    data = ds.argo.canyon_b.create_canyonb_input_matrix()

    assert data.shape[1] == 8


def test_adjust_arctic_latitude(ds, mocked_erddapserver):
    """Test Arctic latitude adjustment"""
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


def test_load_weights(ds, mocked_erddapserver):
    """Test loading of CANYON-B weights"""
    for param in ["AT", "DIC", "pHT"]:
        weights = ds.argo.canyon_b.load_weights(param)
        assert hasattr(weights, "shape")
        assert weights.shape[0] > 0
        assert weights.shape[1] > 0


def test_decimal_year(ds, mocked_erddapserver):
    """Test decimal year calculation"""
    dec_year = ds.argo.canyon_b.decimal_year
    
    # Check it has same length as data
    assert len(dec_year) == len(ds.argo.canyon_b._obj[ds.argo.canyon_b._argo._TNAME])


@pytest.mark.parametrize(
      "param,expected_unit",
      [
          ("PO4", "micromole/kg"),
          ("pHT", "insitu total scale"),
          ("pCO2", "micro atm"),
      ],
  )
def test_get_param_attrs(ds, param, expected_unit, mocked_erddapserver):
    """Test parameter attributes are correctly defined"""
    attrs = ds.argo.canyon_b.get_param_attrs(param)

    assert 'units' in attrs
    assert 'long_name' in attrs
    assert 'comment' in attrs
    assert 'reference' in attrs
    assert attrs['units'] == expected_unit


@pytest.mark.parametrize(
    "use_single_point,include_uncertainties,param",
    [
        (False, True, "PO4"),   # Multi-point with uncertainties
        (False, True, "AT"),    # Multi-point with uncertainties
        (True, True, "AT"),     # Single-point with uncertainties
    ],
)
def test_predict_variations(ds, use_single_point, include_uncertainties, param, mocked_erddapserver):
    """Test CANYON-B predictions with different dataset sizes and uncertainty options"""
    if use_single_point:
        ds = ds.where(ds['N_POINTS'] == 1, drop=True)
        expected_size = 1
    else:
        expected_size = ds.argo.N_POINTS

    ds_result = ds.argo.canyon_b.predict(param, include_uncertainties=include_uncertainties)

    assert param in ds_result
    assert ds_result[param].size == expected_size
    
    if include_uncertainties:
        assert f"{param}_ci" in ds_result
        assert f"{param}_cim" in ds_result
        assert f"{param}_cin" in ds_result
        assert f"{param}_cii" in ds_result


def test_validate_against_matlab():
    """Validate CANYON-B predictions against reference Matlab implementation values

    This test uses the reference values from the original Matlab implementation:
    https://github.com/HCBScienceProducts/CANYON-B/blob/a5b1efce24fcffc8b9e6dd3a0d1e54fa384a01d5/CANYONB.m#L43

    Reference case:
    - Date: 09-Dec-2014 08:45
    - Location: 17.6° N, -24.3° E
    - Depth: 180 dbar
    - Temperature: 16 °C
    - Salinity: 36.1 psu
    - Oxygen: 104 µmol O2 kg-1
    """

    def matlab_ref():
        """Create a dataset with Matlab reference values"""
        # Input values
        biblio_input = {
            'TIME': pd.to_datetime('09-Dec-2014 08:45'),
            'LATITUDE': 17.6,
            'LONGITUDE': -24.3,
            'PRES': 180.0,
            'TEMP': 16.0,
            'PSAL': 36.1,
            'DOXY': 104.0
        }

        # Reference predictions from Matlab implementation
        biblio_predict = [
            {'param': 'NO3', 'ref': 17.91522, 'sigma': 1.32494}, # sigma represents the parameter uncertainty
            {'param': 'PO4', 'ref': 1.081163, 'sigma': 0.073566},
            {'param': 'SiOH4', 'ref': 5.969813, 'sigma': 2.485283},
            {'param': 'AT', 'ref': 2359.331, 'sigma': 9.020},
            {'param': 'DIC', 'ref': 2197.927, 'sigma': 9.151},
            {'param': 'pHT', 'ref': 7.866380, 'sigma': 0.022136},
            {'param': 'pCO2', 'ref': 637.0937, 'sigma': 56.5193}
        ]

 
        def da(key, value):
            return xr.DataArray(np.atleast_1d(value), dims='N_POINTS', name=key)

        l = []
        for p in biblio_predict:
            l.append(da(p['param'], p['ref']))
            l.append(da(f"{p['param']}_s", p['sigma']))
        for p in biblio_input:
            l.append(da(p, biblio_input[p]))
        ds = xr.merge(l)
        ds = ds.set_coords(['LATITUDE', 'LONGITUDE', 'TIME'])
        ds['PLATFORM_NUMBER'] = da('PLATFORM_NUMBER', 100000)
        ds['CYCLE_NUMBER'] = da('CYCLE_NUMBER', 1)
        ds['DIRECTION'] = da('DIRECTION', 'A')
        return ds


    # Get reference values:
    dsref = matlab_ref()

    # Make predictions:
    dspredict = matlab_ref().argo.canyon_b.predict()

    results = []
    nsigma_test = 4
    for key in ['NO3', 'PO4', 'SiOH4', 'AT', 'DIC', 'pHT', 'pCO2']:
    #for key in ['NO3', 'AT', 'pCO2']:
        vref, vsigma = dsref[key].item(), dsref[f'{key}_s'].item()
        vpredict = dspredict[key].item()
        results.append({'param': key,
                        'ref': f"{vref:0.4f}",
                        'sigma': f"{vsigma:0.4f}",
                        'canyon-b': f"{vpredict:0.4f}",
                        f'diff<sigma/{nsigma_test}': vpredict < vref + vsigma / nsigma_test and vpredict > vref - vsigma / nsigma_test,
                        'relative diff (%)': 100 * np.abs(vpredict - vref) / vref
                        })
    df = pd.DataFrame(results)
    print(f"CANYON-B predictions validation against Matlab implementation:\n{df}")
    assert np.all(df[f'diff<sigma/{nsigma_test}'])
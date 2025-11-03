import pytest
import logging

import numpy as np
import xarray as xr
import pandas as pd

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver
from utils import requires_pyco2sys

pytestmark = requires_pyco2sys

log = logging.getLogger("argopy.tests.extensions.carbonate_content")
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


def test_get_canyon_b_raw_predictions(fetcher, mocked_erddapserver):
    """Test getting raw CANYON-B predictions"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(
        params=params,
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=0.01
    )

    # Check that all parameters are present
    for param in params:
        assert param in raw_predictions
        assert param in raw_predictions[param] 
        assert f"{param}_ci" in raw_predictions[param]
        assert f"{param}_cim" in raw_predictions[param]
        assert f"{param}_cin" in raw_predictions[param]    
        assert f"{param}_cii" in raw_predictions[param]
        assert f"{param}_inx" in raw_predictions[param]


def test_setup_pre_carbonate_calculations(fetcher, mocked_erddapserver):
    """Test setup of pre-carbonate calculations"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions first
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(
        params=params,
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=0.01
    )

    # Setup pre-carbonate calculations
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions,
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=0.01
    )

    # Check CANYONData structure
    assert hasattr(canyon_data, 'b_raw')
    assert hasattr(canyon_data, 'covariance')
    assert hasattr(canyon_data, 'correlation')

    # Check MeasurementErrors structure
    assert hasattr(errors, 'salinity')
    assert hasattr(errors, 'temperature')

    # Check rawout and sigma have correct structure
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in rawout
        assert rawout[param].shape == (nol, 4)
        assert param in sigma
        assert sigma[param].shape == (nol, 4)

    # Check covariance and correlation matrices have correct shape
    assert canyon_data.covariance.shape == (4, 4, nol)
    assert canyon_data.correlation.shape == (4, 4, nol)


def test_compute_derivatives_carbonate_system(fetcher, mocked_erddapserver):
    """Test computation of carbonate system derivatives"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions and setup
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(params=params)
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions
    )

    # Compute derivatives
    dcout = ds.argo.content.compute_derivatives_carbonate_system(canyon_data=canyon_data)

    # Check shape: (4, 4, 2, nol)
    nol = ds.argo.N_POINTS
    assert dcout.shape == (4, 4, 2, nol)

    # Check that derivatives are not all NaN
    assert not np.all(np.isnan(dcout))

def test_compute_uncertainties_carbonate_system(fetcher, mocked_erddapserver):
    """Test computation of carbonate system uncertainties"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions and setup
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(params=params)
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions
    )

    rawout_updated, sigma_updated = ds.argo.content.compute_uncertainties_carbonate_system(
        canyon_data=canyon_data, errors=errors, rawout=rawout, sigma=sigma
    )

    # Check that all carbonate parameters have updated values and uncertainties
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert rawout_updated[param].shape == (nol, 4)
        assert sigma_updated[param].shape == (nol, 4)
        assert not np.all(np.isnan(rawout_updated[param]))
        assert not np.all(np.isnan(sigma_updated[param]))


def test_compute_weighted_mean_covariance(fetcher, mocked_erddapserver):
    """Test computation of weighted mean covariance"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions and setup
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(params=params)
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions
    )

    # Compute derivatives
    dcout = ds.argo.content.compute_derivatives_carbonate_system(canyon_data=canyon_data)

    # Compute weighted mean covariance
    cocov = ds.argo.content.compute_weighted_mean_covariance(
        dcout=dcout,
        canyon_data=canyon_data,
        sigma=sigma
    )

    # Check structure
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in cocov
        assert cocov[param].shape == (4, 4, nol)


def test_define_weights(fetcher, mocked_erddapserver):
    """Test weight definition based on uncertainties"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions and setup
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(params=params)
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions
    )

    # Define weights
    weights = ds.argo.content.define_weights(sigma=sigma)

    # Check that weights exist for all parameters
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in weights
        assert f"{param}sum" in weights
        assert weights[param].shape == (nol, 4)


def test_compute_weighted_mean_outputs_and_uncertainties(fetcher, mocked_erddapserver):
    """Test computation of weighted mean outputs and uncertainties"""
    ds = fetcher.to_xarray()
    params = ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']

    # Get raw predictions and setup
    raw_predictions = ds.argo.content.get_canyon_b_raw_predictions(params=params)
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=raw_predictions
    )

    # Compute derivatives
    dcout = ds.argo.content.compute_derivatives_carbonate_system(canyon_data=canyon_data)
    rawout, sigma = ds.argo.content.compute_uncertainties_carbonate_system(
        canyon_data=canyon_data, errors=errors, rawout=rawout, sigma=sigma
    )

    # Compute weighted mean covariance
    cocov = ds.argo.content.compute_weighted_mean_covariance(
        dcout=dcout, canyon_data=canyon_data, sigma=sigma
    )

    # Compute weighted mean outputs and uncertainties
    result = ds.argo.content.compute_weighted_mean_outputs_and_uncertainties(
        rawout=rawout, sigma=sigma, cocov=cocov, canyon_data=canyon_data
    )

    # Check structure of output
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert f"{param}_sigma" in result
        assert f"{param}_sigma_min" in result
        assert f"{param}_raw" in result
        assert result[param].shape == (nol,)
        assert not np.all(np.isnan(result[param]))


def test_predict_internal(fetcher, mocked_erddapserver):
    """Test the internal _predict method"""
    ds = fetcher.to_xarray()

    predictions = ds.argo.content._predict(epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01)

    # Check that all expected outputs are present
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in predictions
        assert f"{param}_sigma" in predictions
        assert f"{param}_sigma_min" in predictions
        assert f"{param}_raw" in predictions
        assert predictions[param].shape == (nol,)
        assert not np.all(np.isnan(predictions[param]))


def test_predict_basic(fetcher, mocked_erddapserver):
    """Test basic CONTENT prediction"""
    ds = fetcher.to_xarray()
    result = ds.argo.content.predict()

    # Check that result is an xarray dataset
    assert isinstance(result, xr.Dataset)

    # Check that main parameters are present
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert result[param].shape[0] == ds.argo.N_POINTS


def test_predict_with_custom_errors(fetcher, mocked_erddapserver):
    """Test CONTENT prediction with custom input errors"""
    ds = fetcher.to_xarray()
    result = ds.argo.content.predict(
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=0.01
    )

    # Check that result contains expected outputs
    assert isinstance(result, xr.Dataset)
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert result[param].shape[0] == ds.argo.N_POINTS


def test_predict_with_array_edoxy(fetcher, mocked_erddapserver):
    """Test CONTENT prediction with array edoxy"""
    ds = fetcher.to_xarray()
    nol = ds.argo.N_POINTS
    edoxy_array = np.full(nol, 0.01)

    result = ds.argo.content.predict(
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=edoxy_array
    )

    # Check that result is valid
    assert isinstance(result, xr.Dataset)
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert result[param].shape[0] == ds.argo.N_POINTS


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
    attrs = ds.argo.content.get_param_attrs(param)

    assert 'units' in attrs
    assert 'long_name' in attrs
    assert 'comment' in attrs
    assert 'reference' in attrs
    assert attrs['units'] == expected_unit


@pytest.mark.parametrize(
    "param",
    ["AT", "DIC", "pHT", "pCO2"],
    indirect=False,
)
def test_predict_single_point(fetcher, param, mocked_erddapserver):
    """Test CONTENT prediction for a single-point dataset"""
    ds = fetcher.to_xarray()
    # Select a single point
    ds_single = ds.where(ds['N_POINTS'] == 1, drop=True)

    # Predict
    ds_result = ds_single.argo.content.predict()

    # Check that prediction was added
    assert param in ds_result
    assert ds_result[param].size == 1


@pytest.mark.parametrize(
    "param",
    ["AT", "pCO2"],
    indirect=False,
)
def test_predict_with_uncertainties(fetcher, param, mocked_erddapserver):
    """Test CONTENT prediction with include_uncertainties=True"""
    ds = fetcher.to_xarray()

    # Predict with uncertainties
    ds_result = ds.argo.content.predict(include_uncertainties=True)

    # Check that prediction and uncertainties were added
    assert param in ds_result
    assert f"{param}_SIGMA" in ds_result
    assert f"{param}_SIGMA_MIN" in ds_result


def test_predict_single_point_with_uncertainties(fetcher, mocked_erddapserver):
    """Test CONTENT prediction for a single-point dataset with uncertainties"""
    ds = fetcher.to_xarray()
    # Select a single point
    ds_single = ds.where(ds['N_POINTS'] == 1, drop=True)

    param = "AT"

    # Predict with uncertainties
    ds_result = ds_single.argo.content.predict(include_uncertainties=True)

    # Check that prediction and uncertainties were added
    assert param in ds_result
    assert f"{param}_SIGMA" in ds_result
    assert f"{param}_SIGMA_MIN" in ds_result

    # Check all have size 1
    assert ds_result[param].size == 1
    assert ds_result[f"{param}_SIGMA"].size == 1
    assert ds_result[f"{param}_SIGMA_MIN"].size == 1


def test_validate_against_matlab():
    """Validate CONTENT predictions against reference Matlab implementation values
    
    This test uses the reference values from the original Matlab implementation:
    https://github.com/HCBScienceProducts/CONTENT/blob/9b644d1c61209d2d6f7681e9e1e4864ef1289c0c/CO2CONTENT.m#L45

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
            {'param': 'AT', 'ref': 2357.817, 'sigma': 10.215},
            {'param': 'DIC', 'ref': 2199.472, 'sigma': 9.811},
            {'param': 'pHT', 'ref': 7.870137, 'sigma': 0.021367},
            {'param': 'pCO2', 'ref': 639.8477, 'sigma': 34.1077}
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

    # Make predictions
    dspredict = matlab_ref().argo.content.predict()

    results = []
    nsigma_test = 4
    for key in ['AT', 'DIC', 'pHT', 'pCO2']:
        vref, vsigma = dsref[key].item(), dsref[f'{key}_s'].item()
        vpredict = dspredict[key].item()
        results.append({'param': key,
                        'ref': f"{vref:0.4f}",
                        'sigma': f"{vsigma:0.4f}",
                        'CONTENT': f"{vpredict:0.4f}",
                        f'diff<sigma/{nsigma_test}': vpredict < vref + vsigma / nsigma_test and vpredict > vref - vsigma / nsigma_test,
                        'relative diff (%)': 100 * np.abs(vpredict - vref) / vref
                        })
    df = pd.DataFrame(results)
    print(f"CONTENT predictions validation against Matlab implementation:\n{df}")
    assert np.all(df[f'diff<sigma/{nsigma_test}'])
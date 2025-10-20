import pytest
import logging
import numpy as np
import xarray as xr

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver


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
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in weights
        assert f"{param}sum" in weights
        assert weights[param].shape == sigma[param].shape


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


def test_predict_basic(fetcher, mocked_erddapserver):
    """Test basic CONTENT prediction"""
    ds = fetcher.to_xarray()
    result = ds.argo.content.predict()

    # Check that result is an xarray dataset
    assert isinstance(result, xr.Dataset)

    # Check that main parameters are present
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert f"{param}_sigma" in result
        assert f"{param}_sigma_min" in result
        assert f"{param}_raw" in result


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
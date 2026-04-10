import pytest
import logging

import numpy as np
import xarray as xr
import pandas as pd

from argopy import DataFetcher
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver
from utils import requires_pyco2sys, requires_joblib

pytestmark = [requires_pyco2sys, requires_joblib]

log = logging.getLogger("argopy.tests.extensions.carbonate_content")
USE_MOCKED_SERVER = True

#
# Define fixtures for the various steps in the CONTENT prediction pipeline
#

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
    """Base dataset - computed once per test module"""
    return fetcher.to_xarray()


@pytest.fixture
def params():
    """Parameters needed for CANYON-B predictions"""
    return ['AT', 'DIC', 'pHT', 'pCO2', 'PO4', 'SiOH4']


@pytest.fixture
def error_params():
    """Error parameters"""
    return {'epres': 0.5, 'etemp': 0.005, 'epsal': 0.005, 'edoxy': 0.01}


@pytest.fixture
def canyon_b_raw(ds, params, error_params):
    """CANYON-B raw predictions"""
    return ds.argo.content.get_canyon_b_raw_predictions(
        params=params, **error_params
    )


@pytest.fixture
def pre_carbonate_calculations(ds, canyon_b_raw, error_params):
    """Pre-carbonate calculations"""
    canyon_data, errors, rawout, sigma = ds.argo.content.setup_pre_carbonate_calculations(
        canyonb_results=canyon_b_raw, **error_params
    )
    return {
        'canyon_data': canyon_data,
        'errors': errors,
        'rawout': rawout,
        'sigma': sigma
    }


@pytest.fixture
def derivatives(ds, pre_carbonate_calculations):
    """Carbonate system derivatives"""
    return ds.argo.content.compute_derivatives_carbonate_system(
        canyon_data=pre_carbonate_calculations['canyon_data']
    )


@pytest.fixture
def uncertainties(ds, pre_carbonate_calculations):
    """Carbonate system uncertainties"""
    rawout, sigma = ds.argo.content.compute_uncertainties_carbonate_system(
        canyon_data=pre_carbonate_calculations['canyon_data'],
        errors=pre_carbonate_calculations['errors'],
        rawout=pre_carbonate_calculations['rawout'],
        sigma=pre_carbonate_calculations['sigma']
    )
    return {'rawout': rawout, 'sigma': sigma}


@pytest.fixture
def weighted_covariance(ds, pre_carbonate_calculations, derivatives, uncertainties):
    """Weighted mean covarianc"""
    return ds.argo.content.compute_weighted_mean_covariance(
        dcout=derivatives,
        canyon_data=pre_carbonate_calculations['canyon_data'],
        sigma=uncertainties['sigma']
    )


@pytest.fixture
def internal_prediction(ds, error_params):
    """Internal prediction"""
    return ds.argo.content._predict(**error_params)

#
# Now, we can test pre-computed results
#

def test_get_canyon_b_raw_predictions(canyon_b_raw, params):
    """Test getting raw CANYON-B predictions"""
    # Check that all parameters are present
    for param in params:
        assert param in canyon_b_raw
        assert param in canyon_b_raw[param]
        assert f"{param}_ci" in canyon_b_raw[param]
        assert f"{param}_cim" in canyon_b_raw[param]
        assert f"{param}_cin" in canyon_b_raw[param]
        assert f"{param}_cii" in canyon_b_raw[param]
        assert f"{param}_inx" in canyon_b_raw[param]


def test_setup_pre_carbonate_calculations(ds, pre_carbonate_calculations):
    """Test setup of pre-carbonate calculations"""
    canyon_data = pre_carbonate_calculations['canyon_data']
    errors = pre_carbonate_calculations['errors']
    rawout = pre_carbonate_calculations['rawout']
    sigma = pre_carbonate_calculations['sigma']

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


def test_compute_derivatives_carbonate_system(ds, derivatives):
    """Test computation of carbonate system derivatives"""
    # Check shape: (4, 4, 2, nol)
    nol = ds.argo.N_POINTS
    assert derivatives.shape == (4, 4, 2, nol)

    # Check that derivatives are not all NaN
    assert not np.all(np.isnan(derivatives))


def test_compute_uncertainties_carbonate_system(ds, uncertainties):
    """Test computation of carbonate system uncertainties"""
    rawout_updated = uncertainties['rawout']
    sigma_updated = uncertainties['sigma']

    # Check that all carbonate parameters have updated values and uncertainties
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert rawout_updated[param].shape == (nol, 4)
        assert sigma_updated[param].shape == (nol, 4)
        assert not np.all(np.isnan(rawout_updated[param]))
        assert not np.all(np.isnan(sigma_updated[param]))


def test_compute_weighted_mean_covariance(ds, weighted_covariance):
    """Test computation of weighted mean covariance"""
    cocov = weighted_covariance

    # Check structure
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in cocov
        assert cocov[param].shape == (4, 4, nol)


def test_define_weights(ds, pre_carbonate_calculations):
    """Test weight definition based on uncertainties"""
    sigma = pre_carbonate_calculations['sigma']

    # Define weights
    weights = ds.argo.content.define_weights(sigma=sigma)

    # Check that weights exist for all parameters
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in weights
        assert f"{param}sum" in weights
        assert weights[param].shape == (nol, 4)


def test_compute_weighted_mean_outputs_and_uncertainties(
    ds, pre_carbonate_calculations, uncertainties, weighted_covariance
):
    """Test computation of weighted mean outputs and uncertainties"""
    # Compute weighted mean outputs and uncertainties
    result = ds.argo.content.compute_weighted_mean_outputs_and_uncertainties(
        rawout=uncertainties['rawout'],
        sigma=uncertainties['sigma'],
        cocov=weighted_covariance,
        canyon_data=pre_carbonate_calculations['canyon_data']
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


def test_predict_internal(ds, internal_prediction):
    """Test the internal _predict method"""
    predictions = internal_prediction

    # Check that all expected outputs are present
    nol = ds.argo.N_POINTS
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in predictions
        assert f"{param}_sigma" in predictions
        assert f"{param}_sigma_min" in predictions
        assert f"{param}_raw" in predictions
        assert predictions[param].shape == (nol,)
        assert not np.all(np.isnan(predictions[param]))


@pytest.mark.parametrize(
    "edoxy_type",
    ["scalar", "array"],
)
def test_predict_with_custom_errors(ds, edoxy_type, mocked_erddapserver):
    """Test CONTENT prediction with custom input errors (scalar and array)"""
    if edoxy_type == "scalar":
        edoxy = 0.01
    else: 
        nol = ds.argo.N_POINTS
        edoxy = np.full(nol, 0.01)

    result = ds.argo.content.predict(
        epres=0.5,
        etemp=0.005,
        epsal=0.005,
        edoxy=edoxy
    )

    # Check that result contains expected outputs
    assert isinstance(result, xr.Dataset)
    for param in ['AT', 'DIC', 'pHT', 'pCO2']:
        assert param in result
        assert result[param].shape[0] == ds.argo.N_POINTS


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
    attrs = ds.argo.content.get_param_attrs(param)

    assert 'units' in attrs
    assert 'long_name' in attrs
    assert 'comment' in attrs
    assert 'reference' in attrs
    assert attrs['units'] == expected_unit


@pytest.mark.parametrize(
    "use_single_point,include_uncertainties,param",
    [
        (False, False, "AT"),  # Multi-point without uncertainties
        (False, True, "AT"),   # Multi-point with uncertainties
        (True, False, "AT"),   # Single-point without uncertainties
        (True, True, "AT"),    # Single-point with uncertainties
    ],
)
def test_predict_variations(ds, use_single_point, include_uncertainties, param, mocked_erddapserver):
    """Test CONTENT predictions with different dataset sizes and uncertainty options"""
    if use_single_point:
        ds = ds.where(ds['N_POINTS'] == 1, drop=True)
        expected_size = 1
    else:
        expected_size = ds.argo.N_POINTS

    ds_result = ds.argo.content.predict(include_uncertainties=include_uncertainties)

    # Check that parameter is present
    assert param in ds_result
    assert ds_result[param].size == expected_size

    # Check uncertainties if included
    if include_uncertainties:
        assert f"{param}_SIGMA" in ds_result
        assert f"{param}_SIGMA_MIN" in ds_result
        assert ds_result[param].shape == ds_result[f"{param}_SIGMA"].shape
        assert ds_result[param].shape == ds_result[f"{param}_SIGMA_MIN"].shape


def test_validate_against_matlab():
    """Validate CONTENT predictions against reference Matlab implementation values

    This test uses reference values to verify the CONTENT method produces
    consistent results with the original Matlab implementation.
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

        # Reference predictions from Matlab implementation for CONTENT
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

    # Get reference values
    dsref = matlab_ref()

    # Make predictions
    dspredict = matlab_ref().argo.content.predict()

    results = []
    nsigma_test = 4
    for key in ['AT', 'DIC', 'pHT', 'pCO2']:
        vref, vsigma = dsref[key].item(), dsref[f'{key}_s'].item()
        vpredict = dspredict[key].item()
        results.append({
            'param': key,
            'ref': f"{vref:0.4f}",
            'sigma': f"{vsigma:0.4f}",
            'content': f"{vpredict:0.4f}",
            f'diff<sigma/{nsigma_test}': vpredict < vref + vsigma / nsigma_test and vpredict > vref - vsigma / nsigma_test,
            'relative diff (%)': 100 * np.abs(vpredict - vref) / vref
        })
    df = pd.DataFrame(results)
    print(f"CONTENT predictions validation against Matlab implementation:\n{df}")
    assert np.all(df[f'diff<sigma/{nsigma_test}'])
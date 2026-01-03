from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np
import xarray as xr

try:
    import PyCO2SYS as pyco2

    HAS_PYCO2SYS = True
except ImportError:
    HAS_PYCO2SYS = False
    pyco2 = None

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Define dummy decorators (needed for tests when numba is not installed)
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Parallel = None
    delayed = None

from ..errors import InvalidDatasetStructure, DataNotFound
from ..utils import path2assets, to_list, point_in_polygon
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor("canyon_b")
class CanyonB(ArgoAccessorExtension):
    """Nutrients and Carbonate System Variables predictor with CANYON-B

    This is an implementation of the CANYON-B method: a bayesian neural network approach that estimates water-column nutrient concentrations and carbonate system variables ([1]_).
    CANYON-B is based on the CANYON model ([2]_) and provides more robust neural networks, that include a local uncertainty estimate for each predicted parameter.

    When using this method, please cite the papers.

    See Also
    --------
    :meth:`canyon_b.predict`, :attr:`canyon_b.input_list`, :attr:`canyon_b.output_list`

    Examples
    --------
    Load data, they must contain oxygen measurements:

    .. code-block:: python

        from argopy import DataFetcher
        ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').float(1902605)
        ds = ArgoSet.to_xarray()

    Once input data are loaded, make all or selected parameters predictions with or without specifying input errors
    on pressure (epres, in dbar), temperature (etemp, in °C), salinity (epsal, in PSU) and oxygen (edoxy, in micromole/kg).
    For interested users, uncertainties on predicted parameters can also be included.

    .. code-block:: python

        ds.argo.canyon_b.predict()
        ds.argo.canyon_b.predict('PO4')
        ds.argo.canyon_b.predict(['PO4', 'NO3'])
        ds.argo.canyon_b.predict(['PO4', 'NO3'], include_uncertainties=True)
        ds.argo.canyon_b.predict(['PO4', 'NO3'], epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01)
        ds.argo.canyon_b.predict(['PO4', 'NO3'], epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01, include_uncertainties=True)

    By default, if no input errors are specified, the following default values are used:
        - epres = 0.5 dbar
        - etemp = 0.005 °C
        - epsal = 0.005 PSU
        - edoxy = 1% of DOXY value

    Notes
    -----
    This Python implementation is largely inspired by work from Raphaël Bajon (https://github.com/RaphaelBajon)
    which is available at https://github.com/RaphaelBajon/canyonbpy and from the EuroGO-SHIP organization (https://github.com/EuroGO-SHIP/AtlantOS_QC/blob/master/atlantos_qc/data_models/extra/pycanyonb.py)

    References
    ----------
    .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. https://doi.org/10.3389/fmars.2018.00328

    .. [2] Sauzède, R., Bittig, H. C., Claustre, H., Pasqueron de Fommervault, O., Gattuso, J. P., Legendre, L., and Johnson, K. S. (2017). Estimates of water-column nutrient concentrations and carbonate system parameters in the global ocean: A novel approach based on neural networks. Frontiers in Marine Science, 4, 128. https://doi.org/10.3389/fmars.2017.00128
    """

    n_inputs = (
        7  # Pressure, Temperature, Salinity, Oxygen, Latitude, Longitude, Decimal Year
    )
    """Number of inputs variables for CANYON-B"""

    _input_list = ["LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL", "DOXY"]
    """List of parameters required to make predictions"""

    _output_list = [
        "PO4",
        "PO4_ci",
        "PO4_cim",
        "PO4_cin",
        "PO4_cii",
        "NO3",
        "NO3_ci",
        "NO3_cim",
        "NO3_cin",
        "NO3_cii",
        "SiOH4",
        "SiOH4_ci",
        "SiOH4_cim",
        "SiOH4_cin",
        "SiOH4_cii",
        "AT",
        "AT_ci",
        "AT_cim",
        "AT_cin",
        "AT_cii",
        "DIC",
        "DIC_ci",
        "DIC_cim",
        "DIC_cin",
        "DIC_cii",
        "pHT",
        "pHT_ci",
        "pHT_cim",
        "pHT_cin",
        "pHT_cii",
        "pCO2",
        "pCO2_ci",
        "pCO2_cim",
        "pCO2_cin",
        "pCO2_cii",
    ]
    # DIC = CT in Bittig et al., (2018), keep it that way to be consistent with the canyon-med extension.
    """List of all possible output variables for CANYON-B"""

    def __init__(self, *args, **kwargs):
        if not HAS_PYCO2SYS:
            raise ImportError(
                "PyCO2SYS is required for the CANYON-B extension."
                "Install it with: pip install PyCO2SYS"
            )
        if not HAS_NUMBA:
            raise ImportError(
                "numba is required for the CANYON-B extension."
                "Install it with: pip install numba"
            )  # Note: for performance reasons, numba is required now.
        if not HAS_JOBLIB:
            raise ImportError(
                "joblib is required for the CANYON-B extension."
                "Install it with: pip install joblib"
            )  # Note: for parallelization of predictions, joblib is required now.

        super().__init__(*args, **kwargs)

        if self._argo._type != "point":
            raise InvalidDatasetStructure(
                "Method only available for a collection of points"
            )
        if self._argo.N_POINTS == 0:
            raise DataNotFound("Empty dataset, no data to transform !")

        self.path2coef = Path(path2assets).joinpath(
            "canyon-b"
        )  # Path to CANYON-B assets

    def get_param_attrs(self, param: str) -> dict:
        """
        Get attributes for a given predicted parameter.

        Parameters
        ----------
        param : str
            Parameter name. Valid options are:

            - 'NO3': Nitrate
            - 'PO4': Phosphate
            - 'SiOH4': Silicate
            - 'AT': Total alkalinity
            - 'DIC': Dissolved inorganic carbon
            - 'pHT': Total pH
            - 'pCO2': Partial pressure of CO2

        Returns
        -------
        dict
            Attribute dictionary containing:

            - 'units': Measurement units
            - 'long_name': Descriptive parameter name
            - 'comment': Data provenance note
            - 'reference': CANYON-B digital object identifier (DOI)
        """
        attrs = {}
        if param in ["NO3", "PO4", "SiOH4", "AT", "DIC"]:
            attrs.update({"units": "micromole/kg"})

        if param == "NO3":
            attrs.update({"long_name": "Nitrate concentration"})

        if param == "PO4":
            attrs.update({"long_name": "Phosphate concentration"})

        if param == "SiOH4":
            attrs.update({"long_name": "Silicate concentration"})

        if param == "AT":
            attrs.update({"long_name": "Total alkalinity"})

        if param == "DIC":
            attrs.update({"long_name": "Total dissolved inorganic carbon"})

        if param == "pHT":
            attrs.update({"long_name": "Total pH"})
            attrs.update({"units": "insitu total scale"})

        if param == "pCO2":
            attrs.update({"long_name": "Partial pressure of CO2"})
            attrs.update({"units": "micro atm"})

        attrs.update({"comment": "Synthetic variable predicted using CANYON-B"})
        attrs.update({"reference": "https://doi.org/10.3389/fmars.2018.00328"})

        return attrs

    @property
    def input_list(self) -> list[str]:
        """List of parameters required to make predictions with CANYON-B

        Returns
        -------
        list[str], default = ["LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL", "DOXY"]
        """
        return self._input_list.copy()

    @property
    def output_list(self) -> list[str]:
        """List of all possible output variables for CANYON-B

        Returns
        -------
        list[str], default = ["PO4", "NO3", "DIC", "SiOH4", "AT", "pHT", "pCO2"]

        Notes
        -----
        DIC = CT in Bittig et al., (2018), keep it that way to be consistent with the canyon-med extension.
        """
        return self._output_list.copy()

    @property
    def decimal_year(self):
        """
        Return the decimal year representation of the dataset `TIME` variable.

        Returns
        -------
        float or np.ndarray
            Decimal year value(s)
        """
        time_array = self._obj[self._argo._TNAME]
        return time_array.dt.year + (
            86400 * time_array.dt.dayofyear
            + 3600 * time_array.dt.hour
            + time_array.dt.second
        ) / (365.0 * 24 * 60 * 60)

    def ds2df(self) -> pd.DataFrame:
        """
        Convert xarray Dataset to CANYON-B neural network input format.

        Transforms the Argo xarray Dataset into a pandas DataFrame with
        the required input variables for the CANYON-B neural network. Applies
        Arctic latitude adjustments and modifies pressure values to account for
        the large range of pressure values (from the surface to 4000 m depth) and
        a non-homogeneous data distribution within this range, according to [1]_.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:

            - 'lat': Latitude in degrees North (Arctic-adjusted if applicable)
            - 'lon': Longitude in degrees East
            - 'dec_year': Decimal year
            - 'temp': Temperature (°C)
            - 'psal': Salinity (PSU)
            - 'doxy': Dissolved oxygen (µmol/kg)
            - 'pres': Modified pressure for CANYON-B input (dimensionless)

        References
        ----------
        .. [1] Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2020). A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 7. https://doi.org/10.3389/fmars.2020.00620
        """
        if self._obj.argo.N_POINTS > 1:
            df = pd.DataFrame(
                {
                    "lat": self.adjust_arctic_latitude(
                        self._obj["LATITUDE"].values, self._obj["LONGITUDE"].values
                    ),
                    "lon": self._obj["LONGITUDE"],
                    "dec_year": self.decimal_year,
                    "temp": self._obj["TEMP"],
                    "psal": self._obj["PSAL"],
                    "doxy": self._obj["DOXY"],
                    "pres": self._obj["PRES"],
                }
            )
        else:  # Handle single point dataset:
            df = pd.DataFrame.from_dict(
                {
                    "lat": self.adjust_arctic_latitude(
                        self._obj["LATITUDE"].values, self._obj["LONGITUDE"].values
                    ),
                    "lon": self._obj["LONGITUDE"].values,
                    "dec_year": self.decimal_year.values,
                    "temp": self._obj["TEMP"].values,
                    "psal": self._obj["PSAL"].values,
                    "doxy": self._obj["DOXY"].values,
                    "pres": self._obj["PRES"].values,
                },
                orient="index",
            ).T

        # Modify pressure according to Eq. 3 in 10.3389/fmars.2020.00620
        df["pres"] = df["pres"].apply(
            lambda x: (x / 2e4) + (1 / ((1 + np.exp(-x / 300)) ** 3))
        )

        return df

    def create_canyonb_input_matrix(self) -> np.ndarray:
        """
        Create input matrix for CANYON-B neural network predictions.

        Converts the xarray Dataset into a numpy array formatted for CANYON-B
        neural network input.

        Returns
        -------
        np.ndarray
            Input matrix containing columns:

            - Decimal year
            - Normalized latitude
            - Transformed longitude (see eq. (1) in [1]_)
            - Temperature (°C)
            - Practical salinity (PSU)
            - Dissolved oxygen (μmol/kg)
            - Transformed pressure (dimensionless)

        References
        ----------
        .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. https://doi.org/10.3389/fmars.2018.00328
        """
        df = self.ds2df()

        # Create input matrix
        data = np.column_stack(
            [
                df["dec_year"],
                df["lat"] / 90,
                np.abs(1 - np.mod(df["lon"] - 110, 360) / 180),
                np.abs(1 - np.mod(df["lon"] - 20, 360) / 180),
                df["temp"],
                df["psal"],
                df["doxy"],
                df["pres"],
            ]
        )

        return data

    def adjust_arctic_latitude(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Adjust latitude for Arctic basin calculations.

        This methods adjusts the latitude of all points inside the Arctic, west of the
        Lomonosov ridge. This adjustment improves the predictions in the subpolar North
        Pacific by artificially increasing the "length" of the Bering Strait (see [1]_ for details).

        Parameters
        ----------
        lat : np.ndarray
            Latitudes in degrees North
        lon : np.ndarray
            Longitudes in degrees East

        Returns
        -------
        np.ndarray
            Adjusted latitudes in degrees North

        References
        ----------
        .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. https://doi.org/10.3389/fmars.2018.00328
        """
        # Points for Arctic basin 'West' of Lomonosov ridge
        plon = np.array(
            [-180, -170, -85, -80, -37, -37, 143, 143, 180, 180, -180, -180]
        )
        plat = np.array([68, 66.5, 66.5, 80, 80, 90, 90, 68, 68, 90, 90, 68])

        # Create polygon array
        polygon = np.column_stack((plon, plat))

        # Flatten coordinates and create points array
        points = np.column_stack((lon.flatten(), lat.flatten()))

        # Get mask
        mask = np.array([point_in_polygon(p, polygon) for p in points])
        mask = mask.reshape(lat.shape)

        # Adjust latitude
        adjusted_lat = lat.copy()
        adjusted_lat[mask] = (
            lat[mask] - np.sin(np.radians(lon[mask] + 37)) * (90 - lat[mask]) * 0.5
        )

        return adjusted_lat

    def load_weights(self, param: str) -> pd.DataFrame:
        """
        Load CANYON-B neural network weights for a specific parameter.

        Parameters
        ----------
        param : str
            Parameter name. Valid options are:

            - 'NO3': Nitrate
            - 'PO4': Phosphate
            - 'SiOH4': Silicate
            - 'AT': Total alkalinity
            - 'DIC': Dissolved inorganic carbon
            - 'pHT': Total pH
            - 'pCO2': Partial pressure of CO2

        Returns
        -------
        pd.DataFrame
            DataFrame containing the neural network weights for the specified parameter.
        """

        if param in ["AT", "pCO2", "NO3", "PO4", "SiOH4"]:
            weights = pd.read_csv(
                self.path2coef.joinpath(f"wgts_{param}.txt"), header=None, sep="\t"
            )
        elif param == "DIC":
            weights = pd.read_csv(
                self.path2coef.joinpath("wgts_CT.txt"), header=None, sep="\t"
            )
        else:
            weights = pd.read_csv(
                self.path2coef.joinpath("wgts_pH.txt"), header=None, sep="\t"
            )

        return weights

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _nn_forward_1layer(data_N, w1, b1, w2, b2):
        """Forward pass for 1-layer neural network (numba optimized with parallelization)"""
        nol = data_N.shape[0]  # Number of data points
        ni = data_N.shape[1]  # Number of inputs to the neural network
        nl1 = w1.shape[0]  # Number of neurons in the hidden layer

        # Forward pass
        a = np.zeros((nol, nl1))
        for i in prange(
            nol
        ):  # Parallel over data points (needs to be an explicit loop for numba)
            for j in range(nl1):
                tmp = b1[j]
                for k in range(ni):
                    tmp += data_N[i, k] * w1[j, k]
                a[i, j] = np.tanh(tmp)

        y = (
            a @ w2.T + b2
        )  # @ is matrix multiplication operator in numpy and numba optimizes it well

        # Calculate input effects in parallel
        inx = np.zeros((nol, ni))
        for i in prange(nol):  # Parallel loop
            tanh_a = a[i, :]
            dtanh = 1 - tanh_a * tanh_a
            for k in range(ni):
                tmp = 0.0
                for j in range(nl1):
                    tmp += w2[0, j] * w1[j, k] * dtanh[j]
                inx[i, k] = tmp

        return y.flatten(), inx

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True, fastmath=True)
    def _nn_forward_2layer(data_N, w1, b1, w2, b2, w3, b3):
        """Forward pass for 2-layer neural network (numba optimized with parallelization)"""
        nol = data_N.shape[0]  # Number of data points
        ni = data_N.shape[1]  # Number of inputs (neural network)
        nl1 = w1.shape[0]  # Number of neurons in the first hidden layer
        nl2 = w2.shape[0]  # Number of neurons in the second hidden layer

        # First layer
        a = np.zeros((nol, nl1))
        for i in prange(nol):
            for j in range(nl1):
                tmp = b1[j]
                for k in range(ni):
                    tmp += data_N[i, k] * w1[j, k]
                a[i, j] = np.tanh(tmp)

        # Second layer
        b_layer = np.zeros((nol, nl2))
        for i in prange(nol):
            for j in range(nl2):
                tmp = b2[j]
                for k in range(nl1):
                    tmp += a[i, k] * w2[j, k]
                b_layer[i, j] = np.tanh(tmp)

        # Output layer
        y = b_layer @ w3.T + b3

        # Calculate input effects in parallel
        inx = np.zeros((nol, ni))
        for i in prange(nol):
            dtanh_a = 1 - a[i, :] * a[i, :]
            dtanh_b = 1 - b_layer[i, :] * b_layer[i, :]

            for m in range(ni):
                tmp = 0.0
                for j in range(nl2):
                    for k in range(nl1):
                        tmp += w3[0, j] * dtanh_b[j] * w2[j, k] * dtanh_a[k] * w1[k, m]
                inx[i, m] = tmp

        return y.flatten(), inx

    def _predict(
        self,
        param: str,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        data: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Predict a single biogeochemical parameter using CANYON-B neural networks.

        This private method implements the CANYON-B Bayesian neural network ensemble
        to estimate oceanic parameters from hydrographic data. It processes input data
        through multiple neural networks and combines their predictions with uncertainty
        estimates.

        Parameters
        ----------
        param : str
            Parameter to predict. Must be one of:
            - 'AT': Total alkalinity (μmol/kg)
            - 'DIC': Dissolved inorganic carbon (μmol/kg)
            - 'pHT': Total pH
            - 'pCO2': Partial pressure of CO₂ (μatm)
            - 'NO3': Nitrate concentration (μmol/kg)
            - 'PO4': Phosphate concentration (μmol/kg)
            - 'SiOH4': Silicate concentration (μmol/kg)
        epres : float, optional
            Pressure measurement uncertainty in dbar (default: 0.5 dbar)
        etemp : float, optional
            Temperature measurement uncertainty in °C (default: 0.005 °C)
        epsal : float, optional
            Salinity measurement uncertainty (PSU, default: 0.005)
        edoxy : float or np.ndarray, optional
            Oxygen measurement uncertainty in μmol/kg. If not provided,
            defaults to 1% of measured oxygen values. Can be a scalar
            applied to all points or an array matching data dimensions.
        data : np.ndarray, optional
            Precomputed input matrix from create_canyonb_input_matrix().
            If not provided, it will be computed.

        Returns
        -------
        dict
            Dictionary containing the predicted parameter and associated uncertainties:
            - param: Predicted parameter value
            - param_ci: Predicted parameter value uncertainty
            - param_cim: Parameter measurement uncertainty
            - param_cin: Parameter uncertainty for Bayesian neural network mapping
            - param_cii: Parameter uncertainty due to input errors
            - param_inx: Input effects on parameter
        """
        # Get array shape and number of elements
        shape = self._obj[self._argo._TNAME].shape
        nol = self._argo.N_POINTS

        # Setting up default uncertainties if not provided
        epres = 0.5 if epres is None else epres
        etemp = 0.005 if etemp is None else etemp
        epsal = 0.005 if epsal is None else epsal
        edoxy = 0.01 * self._obj.DOXY.values if edoxy is None else edoxy

        # Expand scalar error values
        errors = [epres, etemp, epsal, edoxy]
        errors = [
            np.full(nol, e) if np.isscalar(e) else np.asarray(e).flatten()
            for e in errors
        ]
        epres, etemp, epsal, edoxy = errors

        # Define parameters and their properties
        # ! ORDER MATTERS HERE !
        # Order of parameters follows https://github.com/RaphaelBajon/canyonbpy/blob/main/canyonbpy/core.py
        paramnames = ["AT", "DIC", "pHT", "pCO2", "NO3", "PO4", "SiOH4"]
        i_idx = {p: i for i, p in enumerate(paramnames)}
        inputsigma = np.array([6, 4, 0.005, np.nan, 0.02, 0.02, 0.02])
        betaipCO2 = np.array([-3.114e-05, 1.087e-01, -7.899e01])

        # Adjust pH uncertainty (Orr systematic uncertainty)
        inputsigma[2] = np.sqrt(0.005**2 + 0.01**2)

        # Prepare input data
        if data is None:
            data = self.create_canyonb_input_matrix()

        # Output dictionary
        out = {}

        #                                                         #
        # Process through neural networks for the given parameter #
        #                                                         #

        # Get index depending on parameter (mostly important to decipher between nutrients and carbonate systems)
        i = i_idx[param]

        # Load weights and convert them to numpy array
        inwgts = self.load_weights(param)
        inwgts = inwgts.to_numpy()

        # Number of networks in committee
        noparsets = inwgts.shape[1] - 1

        # Determine input normalization based on parameter type
        if i > 3:  # nutrients
            ni = data[:, 1:].shape[1]  # Number of inputs (excluding year)
            ioffset = -1
            mw = inwgts[: ni + 1, -1]
            sw = inwgts[ni + 1 : 2 * ni + 2, -1]
            data_N = (data[:, 1:] - mw[:ni]) / sw[:ni]
        else:  # carbonate system
            ni = data.shape[1]  # Number of inputs
            ioffset = 0
            mw = inwgts[: ni + 1, -1]
            sw = inwgts[ni + 1 : 2 * ni + 2, -1]
            data_N = (data - mw[:ni]) / sw[:ni]

        # Extract weights and prepare arrays
        wgts = inwgts[3, :noparsets]
        betaciw = inwgts[2 * ni + 2 :, -1]
        betaciw = betaciw[~np.isnan(betaciw)]

        # Preallocate arrays
        cval = np.full((nol, noparsets), np.nan)
        cvalcy = np.full(noparsets, np.nan)
        inval = np.full((nol, ni, noparsets), np.nan)

        # Process each network in committee
        for network in range(noparsets):
            nlayerflag = 1 + bool(inwgts[1, network])
            nl1 = int(inwgts[0, network])
            nl2 = int(inwgts[1, network])
            beta = inwgts[2, network]
            # Weights and biases for the first layer
            idx = 4
            w1 = inwgts[idx : idx + nl1 * ni, network].reshape(
                nl1, ni, order="F"
            )  # Here, order='F' is needed to make sure to proper do the calculation as in the Matlab version (https://github.com/HCBScienceProducts/CANYON-B/blob/master/CANYONB.m) !
            idx += nl1 * ni
            b1 = inwgts[idx : idx + nl1, network]
            idx += nl1
            # Weights and biases for the second layer
            w2 = inwgts[idx : idx + nl2 * nl1, network].reshape(nl2, nl1, order="F")
            idx += nl2 * nl1
            b2 = inwgts[idx : idx + nl2, network]

            if nlayerflag == 2:
                # Weights and biases for the third layer (if it exists)
                idx += nl2
                w3 = inwgts[idx : idx + nl2, network].reshape(1, nl2, order="F")
                idx += nl2
                b3 = inwgts[idx : idx + 1, network]

            # Forward pass using numba-optimized functions
            if nlayerflag == 1:
                # One hidden layer
                y, inx = self._nn_forward_1layer(data_N, w1, b1, w2, b2)
            else:
                # Two hidden layers
                y, inx = self._nn_forward_2layer(data_N, w1, b1, w2, b2, w3, b3)

            # Store results
            cval[:, network] = y
            cvalcy[network] = 1 / beta  # 'noise' variance
            inval[:, :, network] = inx

        # Denormalization
        cval = cval * sw[ni] + mw[ni]
        cvalcy = cvalcy * sw[ni] ** 2

        # Calculate committee statistics
        V1 = np.sum(wgts)
        V2 = np.sum(wgts**2)
        pred = np.sum(wgts[None, :] * cval, axis=1) / V1

        # Calculate uncertainties
        cvalcu = np.sum(wgts[None, :] * (cval - pred[:, None]) ** 2, axis=1) / (
            V1 - V2 / V1
        )
        cvalcib = np.sum(wgts * cvalcy) / V1
        cvalciw = np.polyval(betaciw, np.sqrt(cvalcu)) ** 2

        # Calculate input effects
        inx = np.sum(wgts[None, None, :] * inval, axis=2) / V1
        inx = np.tile((sw[ni] / sw[0:ni].T), (nol, 1)) * inx

        # Pressure scaling
        pres_original = self._obj["PRES"].values.flatten()
        ddp = (
            1 / 2e4
            + 1
            / ((1 + np.exp(-pres_original / 300)) ** 4)
            * np.exp(-pres_original / 300)
            / 100
        )
        inx[:, 7 + ioffset] *= ddp

        # Calculate input variance
        error_matrix = np.column_stack([etemp, epsal, edoxy, epres])
        cvalcin = np.sum(
            inx[:, 4 + ioffset : 8 + ioffset] ** 2 * error_matrix**2, axis=1
        )

        # Calculate measurement uncertainty
        if i > 3:
            cvalcimeas = (inputsigma[i] * pred) ** 2
        elif i == 3:
            cvalcimeas = np.polyval(betaipCO2, pred) ** 2
        else:
            cvalcimeas = inputsigma[i] ** 2

        # Calculate total uncertainty
        uncertainty = np.sqrt(cvalcimeas + cvalcib + cvalciw + cvalcu + cvalcin)

        # Create numpy arrays
        out[param] = np.reshape(pred, shape)
        out[f"{param}_ci"] = np.reshape(uncertainty, shape)
        out[f"{param}_cim"] = np.sqrt(cvalcimeas)
        out[f"{param}_cin"] = np.reshape(np.sqrt(cvalcib + cvalciw + cvalcu), shape)
        out[f"{param}_cii"] = np.reshape(np.sqrt(cvalcin), shape)
        out[f"{param}_inx"] = inx[:, 4 : 8 + ioffset]  # Input effects

        # pCO2
        if i == 3:
            # ipCO2 = 'DIC' / umol kg-1 -> pCO2 / uatm
            outcalc = pyco2.sys(
                par1=2300,
                par2=out[param],
                par1_type=1,
                par2_type=2,
                salinity=35.0,
                temperature=25.0,
                temperature_out=np.nan,
                pressure_out=0.0,
                pressure_atmosphere_out=np.nan,
                total_silicate=0.0,
                total_phosphate=0.0,
                opt_pH_scale=1.0,
                opt_k_carbonic=10.0,
                opt_k_bisulfate=1.0,
                grads_of=["pCO2"],
                grads_wrt=["par2"],
            )

            out[f"{paramnames[i]}"] = outcalc["pCO2"]

            # epCO2 = dpCO2/dDIC * e'DIC'
            for unc in ["_ci", "_cin", "_cii"]:
                out[f"{param}{unc}"] = outcalc["d_pCO2__d_par2"] * out[f"{param}{unc}"]

            out[f"{param}_cim"] = outcalc["d_pCO2__d_par2"] * np.reshape(
                out[f"{param}_cim"], shape
            )

            out[f"{param}_inx"] = (
                outcalc["d_pCO2__d_par2"][:, None] * out[f"{param}_inx"]
            )

        return out

    def predict(
        self,
        params: Union[str, List[str]] = None,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        include_uncertainties: Optional[bool] = False,
        n_jobs: Optional[int] = -1,
    ) -> xr.Dataset:
        """
        Make predictions using the CANYON-B method.

        This method implements the CANYON-B Bayesian neural network ensemble
        to estimate oceanic parameters from hydrographic data.

        Parameters
        ----------
        params : str, list of str, or None, optional
            Parameter(s) to predict. Valid options:

            - 'AT': Total alkalinity (μmol/kg)
            - 'DIC': Dissolved inorganic carbon (μmol/kg)
            - 'pHT': Total pH
            - 'pCO2': Partial pressure of CO₂ (μatm)
            - 'NO3': Nitrate concentration (μmol/kg)
            - 'PO4': Phosphate concentration (μmol/kg)
            - 'SiOH4': Silicate concentration (μmol/kg)

            If None (default), all seven parameters are predicted.

        epres : float, optional
            Pressure measurement uncertainty in dbar (default: 0.5 dbar)
        etemp : float, optional
            Temperature measurement uncertainty in °C (default: 0.005 °C)
        epsal : float, optional
            Salinity measurement uncertainty in PSU (default: 0.005)
        edoxy : float or np.ndarray, optional
            Oxygen measurement uncertainty in μmol/kg. If not provided,
            defaults to 1% of measured oxygen values. Can be a scalar
            applied to all points or an array matching data dimensions.
        include_uncertainties : bool, optional
            If True, include uncertainty estimates for each predicted parameter
        n_jobs : int, optional
            Number of parallel jobs used for prediction (only used when there is more than one parameter to predict).
            Default is -1 (use all available CPUs). This option is directly passed to :class:`joblib.Parallel`.

        Returns
        -------
        :class:`xr.Dataset`
            Input dataset augmented with predicted parameters.
        """

        # Validation of requested parameters to predict:
        params_list = ["NO3", "PO4", "SiOH4", "AT", "DIC", "pHT", "pCO2"]
        if params is None:
            params = params_list
        else:
            params = to_list(params)
        for p in params:
            if p not in params_list:
                raise ValueError(
                    "Invalid parameter ('%s') to predict, must be in [%s]"
                    % (p, ",".join(params_list))
                )

        # Compute input matrix once for all parameters (optimization)
        data = self.create_canyonb_input_matrix()

        # Helper function to process a single parameter
        def process_param(param):
            """Process a single parameter prediction"""
            out = self._predict(
                param, epres=epres, etemp=etemp, epsal=epsal, edoxy=edoxy, data=data
            )
            return param, out

        # Make predictions of each of the requested parameters
        if len(params) > 1:
            # Parallel execution
            with Parallel(n_jobs=n_jobs, prefer=None) as parallel:
                results = parallel(delayed(process_param)(param) for param in params)

            # Convert results list to dict for processing
            results_dict = {param: out for param, out in results}
        else:
            # Sequential execution
            results_dict = {param: process_param(param)[1] for param in params}

        # Add results to dataset
        for param in params:
            out = results_dict[param]

            # Add predicted parameter to xr.Dataset
            self._obj[param] = xr.zeros_like(self._obj["TEMP"])
            self._obj[param].attrs = self.get_param_attrs(param)
            values = out[param].astype(np.float32).squeeze()
            self._obj[param].values = np.atleast_1d(values)

            # Add uncertainties if requested
            if include_uncertainties:
                # Uncertainty on parameter (ci)
                self._obj[f"{param}_ci"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_ci"].attrs = self.get_param_attrs(param)
                self._obj[f"{param}_ci"].attrs[
                    "long_name"
                ] = f"Uncertainty on {self.get_param_attrs(param)['long_name']}"
                values = out[f"{param}_ci"].astype(np.float32).squeeze()
                self._obj[f"{param}_ci"].values = np.atleast_1d(values)

                # Measurement uncertainty (cim)
                cim_value = out[f"{param}_cim"]
                if np.isscalar(cim_value):
                    self._obj[f"{param}_cim"] = xr.full_like(
                        self._obj[param], fill_value=float(cim_value), dtype=np.float32
                    )
                else:
                    self._obj[f"{param}_cim"] = xr.zeros_like(self._obj[param])
                    values = cim_value.astype(np.float32).squeeze()
                    self._obj[f"{param}_cim"].values = np.atleast_1d(values)
                self._obj[f"{param}_cim"].attrs = self.get_param_attrs(param)
                self._obj[f"{param}_cim"].attrs[
                    "long_name"
                ] = f"Measurement uncertainty on {self.get_param_attrs(param)['long_name']}"

                # Uncertainty for Bayesian neural network mapping (cin)
                self._obj[f"{param}_cin"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_cin"].attrs = self.get_param_attrs(param)
                self._obj[f"{param}_cin"].attrs[
                    "long_name"
                ] = f"Uncertainty for Bayesian neural network mapping on {self.get_param_attrs(param)['long_name']}"
                values = out[f"{param}_cin"].astype(np.float32).squeeze()
                self._obj[f"{param}_cin"].values = np.atleast_1d(values)

                # Uncertainty due to input errors (cii)
                self._obj[f"{param}_cii"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_cii"].attrs = self.get_param_attrs(param)
                self._obj[f"{param}_cii"].attrs[
                    "long_name"
                ] = f"Uncertainty due to input errors on {self.get_param_attrs(param)['long_name']}"
                values = out[f"{param}_cii"].astype(np.float32).squeeze()
                self._obj[f"{param}_cii"].values = np.atleast_1d(values)

        # Return xr.Dataset with predicted variables:
        if self._argo:
            self._argo.add_history(
                "Added CANYON-B predictions for [%s]" % (",".join(params))
            )

        return self._obj

import numpy as np
import xarray as xr
from typing import Optional, Union

try:
    import PyCO2SYS as pyco2

    HAS_PYCO2SYS = True
except ImportError:
    HAS_PYCO2SYS = False
    pyco2 = None

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Parallel = None
    delayed = None

from ..errors import InvalidDatasetStructure, DataNotFound
from . import register_argo_accessor, ArgoAccessorExtension

# import carbonate utilities
from ..utils.carbonate import (
    error_propagation,
    ChemistryConstants,
    CalculationOptions,
    Measurements,
    MeasurementErrors,
    CANYONData,
)


@register_argo_accessor("content")
class CONTENT(ArgoAccessorExtension):
    """Nutrients and Carbonate System Variables predictor made consistent with chemistry constraints with CONTENT

    This is an implementation of the CONTENT method: a combination of CANYON-B Bayesian neural network mappings of AT, CT, pH and pCO2 made consistent with carbonate chemistry constraints for any set of water column P, T, S, O2, location data as an alternative to (spatial) climatological interpolation ([1]_).

    When using this method, please cite the paper.

    See Also
    --------
    :meth:`content.predict`, :attr:`content.input_list`, :attr:`content.output_list`

    Examples
    --------
    Load data, they must contain oxygen measurements:

    .. code-block:: python

        from argopy import DataFetcher
        # ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').float(1902605)
        ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').region([-75, -60, 20, 30, 0, 1000, '20250101', '20250201'])
        ds = ArgoSet.to_xarray()

    Once input data are loaded, make parameters predictions with or without specifying input errors
    on pressure (epres, in dbar), temperature (etemp, in °C), salinity (epsal, in PSU) and oxygen (edoxy, in micromole/kg).
    For interested users, uncertainties on predicted parameters can also be included.

    .. code-block:: python

        ds.argo.content.predict()
        ds.argo.content.predict(include_uncertainties=True)
        ds.argo.content.predict(epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01)
        ds.argo.content.predict(epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01, include_uncertainties=True)
        ds.argo.content.predict(epres=0.5, etemp=0.005, epsal=0.005, edoxy=np.array([...]))

    By default, if no input errors are specified, the following default values are used:
        - epres = 0.5 dbar
        - etemp = 0.005 °C
        - epsal = 0.005 PSU
        - edoxy = 1% of DOXY value

    Notes
    -----
    This Python implementation is largely inspired by work from HCBSciencesProducts (https://github.com/HCBScienceProducts)
    which is available at https://github.com/HCBScienceProducts/CONTENT. This implementation relies heavily on the great
    PyCO2SYS package (https://github.com/mvdh7/PyCO2SYS) for carbonate chemistry calculations [2]_ which itself is a Python adaptation
    of the original CO2SYS software by C. Lewis and D. Wallace [3_] and subsequent Matlab functions CO2SYS.m by Van Heuven et al. [4]_
    and errors.m and derivnum.m by Orr et al. [5]_.

    References
    ----------

    .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. doi:10.3389/fmars.2018.00328
    .. [2] Humphreys, M. P., Lewis, E. R., Sharp, J. D., & Pierrot, D. (2022). PyCO2SYS v1.8: Marine carbonate system calculations in Python. Geoscientific Model Development, 15(1), 15-43. doi:10.5194/gmd-15-15-2022
    .. [3] Lewis, E. R., & Wallace, D. W. R. (1998). Program developed for CO2 system calculations (No. cdiac: CDIAC-105). Environmental System Science Data Infrastructure for a Virtual Ecosystem (ESS-DIVE)(United States). doi:10.15485/1464255
    .. [4] Van Heuven, S. M. A. C., Pierrot, D., Rae, J. W. B., Lewis, E., & Wallace, D. W. R. (2011). MATLAB program developed for CO2 system calculations. doi: 10.3334/CDIAC/otg.CO2SYS_MATLAB_v1.1
    .. [5] Orr, J. C., Epitalon, J. M., Dickson, A. G., & Gattuso, J. P. (2018). Routine uncertainty propagation for the marine carbon dioxide system. Marine Chemistry, 207, 84-107. doi: 10.1016/j.marchem.2018.10.006
    """

    # Input parameter pairs for each of 6 carbonate calculations (pCO2/AT, pHT/AT, pCO2/pHT, pCO2/DIC, pHT/DIC, AT/DIC)
    _inpar = np.array([[3, 0], [2, 0], [3, 2], [3, 1], [2, 1], [0, 1]])

    # Output parameters (complement of input parameters)
    _outpar = np.array([np.setdiff1d([0, 1, 2, 3], row) for row in _inpar])

    # Flag type of carbonate parameters (used for PyCO2SYS)
    _flag_type = dict(AT=1, DIC=2, pHT=3, pCO2=4)

    # Parameter names used for CONTENT carbonate calculations
    _parameters = ["AT", "DIC", "pHT", "pCO2"]

    # Save indices for output parameters (i.e. _outpar)
    _svi = np.array([[1, 3], [2, 2], [3, 3], [1, 2], [2, 3], [1, 1]])

    _input_list = ["LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL", "DOXY"]
    """List of parameters required to make predictions"""

    _output_list = [
        "PO4",
        "NO3",
        "SiOH4",
        "AT",
        "AT_SIGMA",
        "AT_SIGMA_MIN",
        "DIC",
        "DIC_SIGMA",
        "DIC_SIGMA_MIN",
        "pHT",
        "pHT_SIGMA",
        "pHT_SIGMA_MIN",
        "pCO2",
        "pCO2_SIGMA",
        "pCO2_SIGMA_MIN",
    ]
    """List of all possible output variables for CONTENT"""

    def __init__(self, *args, **kwargs):
        if not HAS_PYCO2SYS:
            raise ImportError(
                "PyCO2SYS is required for the CONTENT extension. "
                "Install it with: pip install PyCO2SYS"
            )

        super().__init__(*args, **kwargs)

        if self._argo._type != "point":
            raise InvalidDatasetStructure(
                "Method only available for a collection of points"
            )
        if self._argo.N_POINTS == 0:
            raise DataNotFound("Empty dataset, no data to transform !")

    @property
    def input_list(self) -> list[str]:
        """List of parameters required to make predictions with CONTENT

        Returns
        -------
        list[str], default = ["LATITUDE", "LONGITUDE", "PRES", "TEMP", "PSAL", "DOXY"]
        """
        return self._input_list.copy()

    @property
    def output_list(self) -> list[str]:
        """List of all possible output variables for CONTENT

        Returns
        -------
        list[str], default = ['PO4', 'NO3', 'SiOH4', 'AT', 'AT_SIGMA', 'AT_SIGMA_MIN', 'DIC', 'DIC_SIGMA', 'DIC_SIGMA_MIN', 'pHT', 'pHT_SIGMA', 'pHT_SIGMA_MIN', 'pCO2', 'pCO2_SIGMA', 'pCO2_SIGMA_MIN']

        Notes
        -----
        DIC = CT in Bittig et al., (2018), keep it that way to be consistent with the canyon-b and canyon-med extensions.
        """
        return self._output_list.copy()

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
            - 'reference': CANYON-B/CONTENT digital object identifier (DOI)
        """
        attrs = {}
        if param in ["NO3", "PO4", "SiOH4", "AT", "DIC"]:
            attrs.update({"units": "micromole/kg"})

        if param == "NO3":
            attrs.update({"long_name": "Nitrate concentration"})
            attrs.update({"comment": "Synthetic variable predicted using CANYON-B"})

        if param == "PO4":
            attrs.update({"long_name": "Phosphate concentration"})
            attrs.update({"comment": "Synthetic variable predicted using CANYON-B"})

        if param == "SiOH4":
            attrs.update({"long_name": "Silicate concentration"})
            attrs.update({"comment": "Synthetic variable predicted using CANYON-B"})

        if param == "AT":
            attrs.update({"long_name": "Total alkalinity"})
            attrs.update({"comment": "Synthetic variable predicted using CONTENT"})

        if param == "DIC":
            attrs.update({"long_name": "Total dissolved inorganic carbon"})
            attrs.update({"comment": "Synthetic variable predicted using CONTENT"})

        if param == "pHT":
            attrs.update({"long_name": "Total pH"})
            attrs.update({"units": "insitu total scale"})
            attrs.update({"comment": "Synthetic variable predicted using CONTENT"})

        if param == "pCO2":
            attrs.update({"long_name": "Partial pressure of CO2"})
            attrs.update({"units": "micro atm"})
            attrs.update({"comment": "Synthetic variable predicted using CONTENT"})

        attrs.update({"reference": "https://doi.org/10.3389/fmars.2018.00328"})

        return attrs

    def get_canyon_b_raw_predictions(
        self,
        params: list,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        n_jobs: Optional[int] = -1,
    ) -> dict:
        """Get raw CANYON-B predictions for specified parameters.

        The raw predicted values contains the predicted parameter values along with their associated uncertainties (ci, cim, cin, cii) and input effects (inx). See `CanyonB._predict` for more details.

        Parameters
        ----------
        params : list
            Parameters that will be predicted. Must be a list containing one or more parameters from 'AT', 'DIC', 'pHT', 'pCO2', 'NO3', 'PO4', 'SiOH4'
        epres, etemp, epsal : float, optional
            Input errors (defaults: 0.5, 0.005, 0.005)
        edoxy : float or np.ndarray, optional
            Oxygen input error (default: 1% of doxy)
        n_jobs : int, optional
            Number of parallel jobs to run (default: -1, use all available CPUs)

        Returns
        -------
        dict
            Dictionary with raw CANYON-B outputs for each parameter

        See Also
        --------
        :class:`Dataset.argo.canyon_b`
        """

        # Compute input matrix once for all parameters (optimization)
        data = self._obj.argo.canyon_b.create_canyonb_input_matrix()

        # Helper function to predict a single parameter
        def predict_param(param):
            """Process a single parameter prediction"""
            out = self._obj.argo.canyon_b._predict(
                param=param,
                epres=epres,
                etemp=etemp,
                epsal=epsal,
                edoxy=edoxy,
                data=data,
            )
            return param, out

        # Get raw predictions for each parameter (in parallel)
        with Parallel(n_jobs=n_jobs, prefer=None) as parallel:
            results = parallel(delayed(predict_param)(param) for param in params)

        # Convert results list to dict
        raw_outputs = {param: out for param, out in results}

        return raw_outputs

    def setup_pre_carbonate_calculations(
        self,
        canyonb_results: dict,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
    ) -> dict:
        """
        Prepare the results from CANYON-B for carbonate calculations.

        This function processes the raw results and sets up the necessary matrices for further carbonate chemistry calculations, returning
        structured data objects.

        Parameters
        ----------
        canyonb_results : dict
            Raw results from CANYON-B predictions.
        epres, etemp, epsal : float, optional
            Input errors (defaults: 0.5, 0.005, 0.005)
        edoxy : float or np.ndarray, optional
            Oxygen input error (default: 1% of doxy)

        Returns
        -------
        canyon_data : CANYONData
          Object with raw predictions, covariance, and correlation
        errors : MeasurementErrors
          Measurement uncertainties for salinity and temperature
        rawout : dict
          Raw output arrays for each carbonate parameter
        sigma : dict
          Uncertainty arrays for each carbonate parameter
        """

        # Number of observations
        nol = self._argo.N_POINTS

        # Setting up default uncertainties if not provided (must match canyon_b._predict defaults)
        epres = 0.5 if epres is None else epres
        etemp = 0.005 if etemp is None else etemp
        epsal = 0.005 if epsal is None else epsal
        edoxy = 0.01 * self._obj.DOXY.values if edoxy is None else edoxy

        # Create MeasurementErrors object (see utils/carbonate.py)
        errors = MeasurementErrors(salinity=epsal, temperature=etemp)

        # Copy to raw output and start mixing calculations
        rawout = {}
        for param in self._parameters:
            rawout[param] = np.full((nol, 4), np.nan)
            rawout[param][:, 0] = canyonb_results[param][param].flatten()

        # Deal with weights and uncertainties
        sigma = {}
        for param in self._parameters:
            sigma[param] = np.full((nol, 4), np.nan)

        # Nutrient uncertainty (variance(inputs) + variance(training data) + variance(MLP)) where MLP is the CANYON-B neural network
        canyonb_results["SiOH4"]["eSiOH4"] = canyonb_results["SiOH4"]["SiOH4_ci"]
        canyonb_results["PO4"]["ePO4"] = canyonb_results["PO4"]["PO4_ci"]

        # covariance matrix of CANYON-B AT, CT, pH, pCO2 due to common inputs
        canyon_cov = np.full((4, 4, nol), np.nan)

        # Create error array
        error_array = [etemp, epsal, edoxy, epres]
        error_array = [
            np.full(nol, e) if np.isscalar(e) else np.asarray(e).flatten()
            for e in error_array
        ]
        etemp, epsal, edoxy, epres = error_array

        # Compute error matrix
        error_matrix = np.column_stack([etemp, epsal, edoxy, epres]) ** 2

        # Construct covariance matrix
        for i in range(4):
            for j in range(i, 4):
                inx_i = canyonb_results[self._parameters[i]][
                    f"{self._parameters[i]}_inx"
                ]
                inx_j = canyonb_results[self._parameters[j]][
                    f"{self._parameters[j]}_inx"
                ]
                canyon_cov[i, j, :] = np.sum(inx_i * inx_j * error_matrix, axis=1)
                canyon_cov[j, i, :] = canyon_cov[i, j, :]

        # Correlation matrix of CANYON-B AT, CT, pH, pCO2 due to common inputs and
        # CANYON-B estimation uncertainty
        cyr = canyon_cov * 0

        # Add full variance on diagonal
        # variance(inputs)[local] + variance(training data)[global] + variance(MLP)[local]
        sigma["AT"][:, 0] = canyonb_results["AT"]["AT_ci"]
        canyon_cov[0, 0, :] = canyonb_results["AT"]["AT_ci"] ** 2

        sigma["DIC"][:, 0] = canyonb_results["DIC"]["DIC_ci"]
        canyon_cov[1, 1, :] = canyonb_results["DIC"]["DIC_ci"] ** 2

        sigma["pHT"][:, 0] = canyonb_results["pHT"]["pHT_ci"]
        canyon_cov[2, 2, :] = canyonb_results["pHT"]["pHT_ci"] ** 2

        sigma["pCO2"][:, 0] = canyonb_results["pCO2"]["pCO2_ci"]
        canyon_cov[3, 3, :] = canyonb_results["pCO2"]["pCO2_ci"] ** 2

        # Convert covariance matrices to correlation matrices
        for i in range(nol):
            # Extract standard deviations from diagonal elements
            std_devs = np.sqrt(np.diag(canyon_cov[:, :, i]))
            # Create normalization matrix (std[i] * std[j] for all i,j pairs)
            std_outer = np.outer(std_devs, std_devs)
            # Convert covariance to correlation: corr[i,j] = cov[i,j] / (std[i] * std[j])
            cyr[:, :, i] = canyon_cov[:, :, i] / std_outer

        # Create CANYONData object
        canyon_data = CANYONData(
            b_raw=canyonb_results, covariance=canyon_cov, correlation=cyr
        )

        return canyon_data, errors, rawout, sigma

    def compute_derivatives_carbonate_system(
        self,
        canyon_data: CANYONData,
        options: CalculationOptions = None,
        n_jobs: Optional[int] = -1,
    ) -> np.ndarray:
        """
        Compute derivatives of carbonate system calculations for all 6 parameter pairs
        (pCO2/AT, pHT/AT, pCO2/pHT, pCO2/DIC, pHT/DIC, AT/DIC).

        Parameters
        ----------
        canyon_data : CANYONData
            CANYON-B predictions with covariance and correlation matrices
        options : CalculationOptions, optional
            PyCO2SYS calculation options (uses defaults if None)
        n_jobs : int, optional
            Number of parallel jobs to run (default: -1, use all available CPUs)

        Returns
        -------
        np.ndarray
            Array containing computed derivatives
        """

        if options is None:
            options = CalculationOptions()

        # Create Measurements object from self._obj data
        measurements = Measurements(
            salinity=self._obj.PSAL.values,
            temperature=self._obj.TEMP.values,
            pressure=self._obj.PRES.values,
            total_silicate=canyon_data.b_raw["SiOH4"]["SiOH4"],
            total_phosphate=canyon_data.b_raw["PO4"]["PO4"],
            total_borate=0.0,  # 0 following CONTENT matlab implementation
        )

        # Number of observations
        nol = self._argo.N_POINTS

        # Initialize output array
        dcout = np.full((4, 4, 2, nol), np.nan)

        # Parameter name mapping for derivatives
        parameters_derived = {
            "AT": {"par1": "d_alkalinity__d_par1", "par2": "d_alkalinity__d_par2"},
            "DIC": {"par1": "d_dic__d_par1", "par2": "d_dic__d_par2"},
            "pHT": {"par1": "d_pH__d_par1", "par2": "d_pH__d_par2"},
            "pCO2": {"par1": "d_pCO2__d_par1", "par2": "d_pCO2__d_par2"},
        }

        # Helper function to compute derivatives for one parameter pair
        def compute_pair_derivatives(p):
            # Get parameter names for this combination
            par1_name = self._parameters[self._inpar[p, 0]]
            par2_name = self._parameters[self._inpar[p, 1]]

            # Get parameter values and types
            par1 = canyon_data.b_raw[par1_name][par1_name]
            par2 = canyon_data.b_raw[par2_name][par2_name]
            par1_type = self._flag_type[par1_name]
            par2_type = self._flag_type[par2_name]

            # Compute derivatives with respect to both par1 and par2
            deriv = pyco2.sys(
                par1=par1,
                par2=par2,
                par1_type=par1_type,
                par2_type=par2_type,
                salinity=measurements.salinity,
                temperature=measurements.temperature,
                pressure=measurements.pressure,
                total_silicate=measurements.total_silicate,
                total_phosphate=measurements.total_phosphate,
                opt_pH_scale=options.pH_scale,
                opt_k_carbonic=options.k_carbonic,
                opt_k_bisulfate=options.k_bisulfate,
                grads_of=["pH", "pCO2", "alkalinity", "dic"],
                grads_wrt=["par1", "par2"],
            )

            # Get output parameter names
            out_param1 = self._parameters[self._outpar[p, 0]]
            out_param2 = self._parameters[self._outpar[p, 1]]

            # Return derivatives for this pair
            return (p, deriv, out_param1, out_param2)

        # Compute derivatives for all 6 parameter pairs (in parallel)
        with Parallel(n_jobs=n_jobs, prefer=None) as parallel:
            results = parallel(delayed(compute_pair_derivatives)(p) for p in range(6))

        # Store results in dcout array
        for p, deriv, out_param1, out_param2 in results:
            # Store derivatives with respect to par1
            dcout[self._inpar[p, 0], self._inpar[p, 1], 0, :] = deriv[
                parameters_derived[out_param1]["par1"]
            ]
            dcout[self._inpar[p, 0], self._inpar[p, 1], 1, :] = deriv[
                parameters_derived[out_param2]["par1"]
            ]

            # Store derivatives with respect to par2
            dcout[self._inpar[p, 1], self._inpar[p, 0], 0, :] = deriv[
                parameters_derived[out_param1]["par2"]
            ]
            dcout[self._inpar[p, 1], self._inpar[p, 0], 1, :] = deriv[
                parameters_derived[out_param2]["par2"]
            ]

        return dcout

    def compute_uncertainties_carbonate_system(
        self,
        canyon_data: CANYONData,
        errors: MeasurementErrors,
        rawout: dict,
        sigma: dict,
        constants: ChemistryConstants = None,
        options: CalculationOptions = None,
        n_jobs: Optional[int] = -1,
    ) -> tuple[dict, dict]:
        """
        Compute carbonate system values and uncertainties for all 6 parameter pairs.
        This function uses the K1K2 constants of Lueker et al., (2000), KSO4 of Dickson 1990 & TB of Uppstrom 1979
        and adds localized error calculations including parameter correlation due to common inputs.

        Parameters
        ----------
        canyon_data : CANYONData
            CANYON-B predictions with covariance and correlation matrices
        errors : MeasurementErrors
            Measurement uncertainties for salinity and temperature
        rawout : dict
            Raw output arrays for each parameter (i.e. AT, DIC, pHT, pCO2)
        sigma : dict
            Uncertainty arrays for each parameter (i.e. AT, DIC, pHT, pCO2)
        constants : ChemistryConstants, optional
            Equilibrium constant uncertainties (uses defaults if None)
        options : CalculationOptions, optional
            PyCO2SYS calculation options (uses defaults if None)
        n_jobs : int, optional
            Number of parallel jobs to run (default: -1, use all available CPUs)

        Returns
        -------
        rawout : dict
            Computed carbonate system values for each parameter (updated)
        sigma : dict
            Uncertainties for each parameter (updated)
        """

        # Set defaults if not provided
        if constants is None:
            constants = ChemistryConstants()
        if options is None:
            options = CalculationOptions()

        # Create Measurements object from self._obj data
        measurements = Measurements(
            salinity=self._obj.PSAL.values,
            temperature=self._obj.TEMP.values,
            pressure=self._obj.PRES.values,
            total_silicate=canyon_data.b_raw["SiOH4"]["SiOH4"],
            total_phosphate=canyon_data.b_raw["PO4"]["PO4"],
            total_borate=0.0,  # 0 following CONTENT matlab implementation
        )

        # Parameter name mapping for output
        parameters_derived = dict(AT="alkalinity", DIC="dic", pHT="pH", pCO2="pCO2")

        # Helper function to compute uncertainties for one parameter combination
        def compute_pair_uncertainties(p):
            # Get parameter types and values for this combination
            par1_name = self._parameters[self._inpar[p, 0]]
            par2_name = self._parameters[self._inpar[p, 1]]

            par1_value = canyon_data.b_raw[par1_name][par1_name]
            par2_value = canyon_data.b_raw[par2_name][par2_name]

            par1_type = self._flag_type[par1_name]
            par2_type = self._flag_type[par2_name]

            # Compute carbonate system with error propagation
            deriv, uncertainties = error_propagation(
                inpar=self._inpar,
                carbonate_index=p,
                parameter_1=par1_value,
                parameter_2=par2_value,
                parameter_1_type=par1_type,
                parameter_2_type=par2_type,
                measurements=measurements,
                errors=errors,
                canyon_data=canyon_data,
                constants=constants,
                options=options,
            )

            # Get output parameter names
            out_param1 = self._parameters[self._outpar[p, 0]]
            out_param2 = self._parameters[self._outpar[p, 1]]

            return (p, deriv, uncertainties, out_param1, out_param2)

        # Compute uncertainties for all 6 parameter combinations (in parallel)
        with Parallel(n_jobs=n_jobs, prefer=None) as parallel:
            results = parallel(delayed(compute_pair_uncertainties)(p) for p in range(6))

        # Store results
        for p, deriv, uncertainties, out_param1, out_param2 in results:
            # Store output values for the two derived parameters
            rawout[out_param1][:, self._svi[p, 0]] = deriv[
                parameters_derived[out_param1]
            ]
            rawout[out_param2][:, self._svi[p, 1]] = deriv[
                parameters_derived[out_param2]
            ]

            # Store uncertainty values
            sigma[out_param1][:, self._svi[p, 0]] = uncertainties[
                parameters_derived[out_param1]
            ]
            sigma[out_param2][:, self._svi[p, 1]] = uncertainties[
                parameters_derived[out_param2]
            ]

        return rawout, sigma

    def compute_weighted_mean_covariance(
        self,
        dcout: np.ndarray,
        canyon_data: CANYONData,
        sigma: np.ndarray,
    ) -> dict:
        """
        Compute weighted mean and covariance for each carbonate parameter.

        Parameters
        ----------
        dcout : np.ndarray
            Derivatives of carbonate system calculations
        canyon_data : CANYONData
            CANYON-B predictions with covariance and correlation matrices
        sigma : np.ndarray
            Uncertainty arrays for each carbonate parameter

        Returns
        -------
        cocov : dict
            Weighted mean covariance matrices for each carbonate parameter
        """

        # Number of observations
        nol = self._argo.N_POINTS

        # Build weighted mean covariance matrix for all parameters
        cocov = {}

        for k in range(4):
            cocov[self._parameters[k]] = np.full((4, 4, nol), np.nan)

            # Fill diagonal with variances
            for i in range(4):
                cocov[self._parameters[k]][i, i, :] = (
                    sigma[self._parameters[k]][:, i] ** 2
                )

            for p in range(6):
                if k in self._outpar[p, :]:  # find calc (p) that calculated parameter k
                    i = np.where(self._outpar[p, :] == k)[0][0]

                    # Covariance from direct CANYON-B and calc (p): inpar[p,:]
                    cocov[self._parameters[k]][0, self._svi[p, i], :] = (
                        1
                        * dcout[self._inpar[p, 0], self._inpar[p, 1], i, :]
                        * canyon_data.covariance[k, self._inpar[p, 0], :]
                        + 1
                        * dcout[self._inpar[p, 1], self._inpar[p, 0], i, :]
                        * canyon_data.covariance[k, self._inpar[p, 1], :]
                    )

                    # Mirror the covariance
                    cocov[self._parameters[k]][self._svi[p, i], 0, :] = cocov[
                        self._parameters[k]
                    ][0, self._svi[p, i], :]

                    # Find second calc (o): inpar[o,:] for covariance term between calc (p) and calc (o)
                    if p < 5:
                        for o in range(p + 1, 6):
                            if (
                                k in self._outpar[o, :]
                            ):  # find calc (o) that calculated parameter k
                                j = np.where(self._outpar[o, :] == k)[0][0]

                                # Covariance from calcs (p) and (o)
                                cocov[self._parameters[k]][
                                    self._svi[p, i], self._svi[o, j], :
                                ] = (
                                    dcout[self._inpar[p, 0], self._inpar[p, 1], i, :]
                                    * dcout[self._inpar[o, 0], self._inpar[o, 1], j, :]
                                    * canyon_data.covariance[
                                        self._inpar[p, 0], self._inpar[o, 0], :
                                    ]
                                    + dcout[self._inpar[p, 0], self._inpar[p, 1], i, :]
                                    * dcout[self._inpar[o, 1], self._inpar[o, 0], j, :]
                                    * canyon_data.covariance[
                                        self._inpar[p, 0], self._inpar[o, 1], :
                                    ]
                                    + dcout[self._inpar[p, 1], self._inpar[p, 0], i, :]
                                    * dcout[self._inpar[o, 0], self._inpar[o, 1], j, :]
                                    * canyon_data.covariance[
                                        self._inpar[p, 1], self._inpar[o, 0], :
                                    ]
                                    + dcout[self._inpar[p, 1], self._inpar[p, 0], i, :]
                                    * dcout[self._inpar[o, 1], self._inpar[o, 0], j, :]
                                    * canyon_data.covariance[
                                        self._inpar[p, 1], self._inpar[o, 1], :
                                    ]
                                )

                                # Mirror the covariance
                                cocov[self._parameters[k]][
                                    self._svi[o, j], self._svi[p, i], :
                                ] = cocov[self._parameters[k]][
                                    self._svi[p, i], self._svi[o, j], :
                                ]
        return cocov

    def define_weights(self, sigma: dict) -> dict:
        """
        Define weights for each carbonate parameter based on uncertainties.

        Parameters
        ----------
        sigma : dict
            Uncertainty arrays for each carbonate parameter

        Returns
        -------
        weights : dict
            Weights for each carbonate parameter
        """

        weights = {}

        for i in range(4):
            weights[self._parameters[i]] = (
                1 / sigma[self._parameters[i]] ** 2
            )  # weights
            weights[f"{self._parameters[i]}sum"] = np.sum(
                weights[self._parameters[i]], axis=1
            )  # sum all to normalize weights to 1

        return weights

    def compute_weighted_mean_outputs_and_uncertainties(
        self, rawout: dict, sigma: dict, cocov: dict, canyon_data: CANYONData
    ) -> dict:
        """
        Compute weighted mean outputs and their uncertainties for each carbonate parameter.

        Parameters
        ----------
        rawout : dict
            Raw output arrays for each carbonate parameter
        sigma : dict
            Uncertainty arrays for each carbonate parameter
        cocov : dict
            Weighted mean covariance matrices for each carbonate parameter
        canyon_data : CANYONData
            CANYON-B predictions with raw results

        Returns
        -------
        out : dict
            Final output dictionary containing:
            - {param}: weighted mean values for each carbonate parameter
            - {param}_sigma: total uncertainty of the CONTENT estimate, that is the sum of sigma_min and sigma_mean where :
                - sigma_mean = standard deviation associated with the weighted mean (of the four carbonate system variable estimates)
                - sigma_min = standard deviation propagated from the uncertainty of the terms of the weighted mean (see Eq. 6 in [1]_)
            - {param}_sigma_min: standard deviation propagated from the uncertainty of the terms of the weighted mean (see Eq. 6 in [1]_)
            - {param}_raw: raw calculation values
            - sigma: record of weighted sigmas
            - canyon_b_raw: CANYON-B structure

        Reference
        ---------
        .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. doi:10.3389/fmars.2018.00328
        """

        # Number of observations
        nol = self._argo.N_POINTS

        # Define weights based on uncertainties
        w = self.define_weights(sigma)

        # Output dictionary
        out = {}

        # Output each variable
        for i in range(4):
            param = self._parameters[i]

            # Weighted mean
            out[param] = (
                np.sum(w[param] * rawout[param], axis=1) / w[f"{param}sum"]
            )  # / [param unit]

            # Standard deviation about the mean (of weighted mean...; for 4 samples.)
            sigma_delta = np.sqrt(
                np.sum(
                    w[param] * (rawout[param] - out[param][:, np.newaxis]) ** 2, axis=1
                )
                / (w[f"{param}sum"] - np.sum(w[param] ** 2, axis=1) / w[f"{param}sum"])
            )  # / [param unit]; is localized

            # Std propagation from correlated inputs for weighted mean
            weight_ratio = w[param] / w[f"{param}sum"][:, np.newaxis]

            # Create the weighted covariance product
            weighted_cov = np.zeros(nol)
            for row in range(4):
                for col in range(4):
                    weighted_cov += (
                        weight_ratio[:, row]
                        * weight_ratio[:, col]
                        * cocov[param][row, col, :]
                    )

            sigma_propagated = np.sqrt(weighted_cov)  # / [param unit]; is localized

            out[f"{param}_sigma"] = sigma_delta + sigma_propagated  # / [param unit]
            out[f"{param}_sigma_min"] = sigma_propagated  # / [param unit]

        # Add raw calculations
        for i in range(4):
            param = self._parameters[i]
            out[f"{param}_raw"] = rawout[param]

        out["sigma"] = sigma  # record of weighted sigmas
        out["canyon_b_raw"] = canyon_data.b_raw  # CANYON-B structure

        return out

    def _predict(
        self,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
    ) -> dict:
        """
        Predict CONTENT variables

        Parameters
        ----------
        epres, etemp, epsal : float, optional
                Input errors
        edoxy : float or np.ndarray, optional
                Oxygen input error (default: 1% of doxy)

        Returns
        -------
        predictions : dict
            Dictionary containing predicted variables with uncertainties
        """

        # Step 1: Get raw CANYON-B predictions for all carbonate parameters + nutrients
        params_to_predict = [
            "AT",
            "DIC",
            "pHT",
            "pCO2",
            "PO4",
            "SiOH4",
            "NO3",
        ]  # Technically, NO3 is not needed for CONTENT but I included it for the final output.
        canyonb_results = self.get_canyon_b_raw_predictions(
            params=params_to_predict, epres=epres, etemp=etemp, epsal=epsal, edoxy=edoxy
        )

        # Step 2: Setup pre-carbonate calculations (covariance, correlation, etc.)
        canyon_data, errors, rawout, sigma = self.setup_pre_carbonate_calculations(
            canyonb_results=canyonb_results,
            epres=epres,
            etemp=etemp,
            epsal=epsal,
            edoxy=edoxy,
        )

        # Step 3: Compute derivatives for carbonate system
        dcout = self.compute_derivatives_carbonate_system(canyon_data=canyon_data)

        # Step 4: Compute uncertainties for carbonate system
        rawout, sigma = self.compute_uncertainties_carbonate_system(
            canyon_data=canyon_data, errors=errors, rawout=rawout, sigma=sigma
        )

        # Step 5: Compute weighted mean covariance
        cocov = self.compute_weighted_mean_covariance(
            dcout=dcout, canyon_data=canyon_data, sigma=sigma
        )

        # Step 6: Compute weighted mean outputs and uncertainties
        results = self.compute_weighted_mean_outputs_and_uncertainties(
            rawout=rawout, sigma=sigma, cocov=cocov, canyon_data=canyon_data
        )

        return results

    def predict(
        self,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
        include_uncertainties: Optional[bool] = False,
    ) -> xr.Dataset:
        """
        Make predictions using the CONTENT method.

        Parameters
        ----------
        epres, etemp, epsal : float, optional
            Input errors
        edoxy : float or np.ndarray, optional
            Oxygen input error (default: 1% of doxy)
        include_uncertainties : bool, optional
            If True, include uncertainty estimates for each predicted parameter

        Returns
        -------
        :class:`xr.Dataset`
            Input dataset augmented with predicted parameters
        """

        # Get predictions for all 4 carbonate parameters (CONTENT method) and nutrients (CANYON-B method)
        prediction = self._predict(epres=epres, etemp=etemp, epsal=epsal, edoxy=edoxy)

        for param in ["NO3", "PO4", "SiOH4"]:
            self._obj[param] = xr.zeros_like(self._obj["TEMP"])
            self._obj[param].attrs = self.get_param_attrs(param)
            values = (
                prediction["canyon_b_raw"][param][param].astype(np.float32).squeeze()
            )
            self._obj[param].values = np.atleast_1d(values)

        # Update history for nutrients
        if self._argo:
            self._argo.add_history(
                "Added CANYON-B predictions for [%s]"
                % (",".join(["NO3", "PO4", "SiOH4"]))
            )

        for param in self._parameters:
            self._obj[param] = xr.zeros_like(self._obj["TEMP"])
            self._obj[param].attrs = self.get_param_attrs(param)
            values = prediction[param].astype(np.float32).squeeze()
            self._obj[param].values = np.atleast_1d(values)

            if include_uncertainties:
                # sigma
                self._obj[f"{param}_SIGMA"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_SIGMA"].attrs = {
                    "long_name": f"Total uncertainty on {param} using CONTENT",
                    "units": self._obj[param].attrs.get("units", ""),
                }
                values = prediction[f"{param}_sigma"].astype(np.float32).squeeze()
                self._obj[f"{param}_SIGMA"].values = np.atleast_1d(values)

                # sigma min
                self._obj[f"{param}_SIGMA_MIN"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_SIGMA_MIN"].attrs = {
                    "long_name": f"Uncertainty propagated from the input variables on {param} using CONTENT",
                    "units": self._obj[param].attrs.get("units", ""),
                }
                values = prediction[f"{param}_sigma_min"].astype(np.float32).squeeze()
                self._obj[f"{param}_SIGMA_MIN"].values = np.atleast_1d(values)

        # Update history for carbonate parameters
        if self._argo:
            self._argo.add_history(
                "Added CONTENT predictions for [%s]" % (",".join(self._parameters))
            )

        # Create and return xarray Dataset with predictions
        return self._obj

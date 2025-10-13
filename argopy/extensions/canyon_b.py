from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np
import xarray as xr
import PyCO2SYS as pyco2

from ..errors import InvalidDatasetStructure, DataNotFound
from ..utils import path2assets, to_list, point_in_polygon
from . import register_argo_accessor, ArgoAccessorExtension


@register_argo_accessor("canyon_b")
class CanyonB(ArgoAccessorExtension):
    """
    Implementation of the CANYON-B method.

    CANYON-B is a bayesian neural network approach that estimate water-column nutrient concentrations
    and carbonate system variables ([1]_). CANYON-B is based on the CANYON model ([2]_) and provides
    more robust neural networks, that include a local uncertainty estimate for each predicted parameter.

    Examples
    --------
    Load data, they must contain oxygen measurements:

    .. code-block:: python

        from argopy import DataFetcher
        ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').float(1902605)
        ds = ArgoSet.to_xarray()

    Once input data are loaded, make all or selected parameters predictions:

    .. code-block:: python

        ds.argo.canyon_b.predict()
        ds.argo.canyon_b.predict('PO4')
        ds.argo.canyon_b.predict(['PO4', 'NO3'])
        ds.argo.canyon_b.predict(['PO4', 'NO3'], epres=0.5, etemp=0.005, epsal=0.005, edoxy=0.01)

    Notes
    -----
    This Python implementation is largely inspired by work from Raphaël Bajon (https://github.com/RaphaelBajon)
    which is available at https://github.com/RaphaelBajon/canyonbpy and from the EuroGO-SHIP organization (https://github.com/EuroGO-SHIP/AtlantOS_QC/blob/master/atlantos_qc/data_models/extra/pycanyonb.py)

    References
    ----------

    .. [1] Bittig, H. C., Steinhoff, T., Claustre, H., Fiedler, B., Williams, N. L., Sauzède, R., Körtzinger, A., and Gattuso, J. P. (2018). An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks. Frontiers in Marine Science, 5, 328. doi:10.3389/fmars.2018.00328

    .. [2] Sauzède, R., Bittig, H. C., Claustre, H., Pasqueron de Fommervault, O., Gattuso, J. P., Legendre, L., and Johnson, K. S. (2017). Estimates of water-column nutrient concentrations and carbonate system parameters in the global ocean: A novel approach based on neural networks. Frontiers in Marine Science, 4, 128. doi:10.3389/fmars.2017.00128

    """

    n_inputs = (
        7  # Pressure, Temperature, Salinity, Oxygen, Latitude, Longitude, Decimal Year
    )
    """Number of inputs variables for CANYON-B"""

    output_list = [
        "AT",
        "DIC",
        "pHT",
        "pCO2",
        "NO3",
        "PO4",
        "SiOH4",
    ]  # DIC = CT in [1], keep it that way to be consistent with the canyon-med extention. Order of parameters follows https://github.com/RaphaelBajon/canyonbpy/blob/main/canyonbpy/core.py because it is used later in the predict method.
    """List of all possible output variables for CANYON-B"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._argo._type != "point":
            raise InvalidDatasetStructure(
                "Method only available for a collection of points"
            )
        if self._argo.N_POINTS == 0:
            raise DataNotFound("Empty dataset, no data to transform !")

        # self.n_list = 5
        self.path2coef = Path(path2assets).joinpath(
            "canyon-b"
        )  # Path to CANYON-B assets
        # self._input = None  # Private CANYON-MED input dataframe

    def get_param_attrs(self, param: str) -> dict:
        """Provides attributes to be added to a given predicted parameter"""
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
    def decimal_year(self):
        """Return the decimal year of the :class:`xr.Dataset` `TIME` variable"""
        time_array = self._obj[self._argo._TNAME]
        return time_array.dt.year + (
            86400 * time_array.dt.dayofyear
            + 3600 * time_array.dt.hour
            + time_array.dt.second
        ) / (365.0 * 24 * 60 * 60)

    def ds2df(self) -> pd.DataFrame:
        """Create a CANYON-B input :class:`pd.DataFrame` from :class:`xr.Dataset`"""

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
                    "lon": self._obj["LONGITUDE"],
                    "dec_year": self.decimal_year,
                    "temp": self._obj["TEMP"],
                    "psal": self._obj["PSAL"],
                    "doxy": self._obj["DOXY"],
                    "pres": self._obj["PRES"],
                },
                orient="index",
            ).T

        # Modify pressure
        # > The pressure input is transformed according to the combination of a linear
        # and a logistic curve to limit the degrees of freedom of the ANN in deep
        # waters and to account for the large range of pressure values (from the
        # surface to 4000 m depth) and a non-homogeneous distribution of data
        # within this range
        # See Eq. 3 in 10.3389/fmars.2020.00620
        df["pres"] = df["pres"].apply(
            lambda x: (x / 2e4) + (1 / ((1 + np.exp(-x / 300)) ** 3))
        )

        return df

    def create_canyonb_input_matrix(self) -> np.ndarray:
        """Create CANYON-B input matrix from :class:`xr.Dataset`"""
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

    # Latitude adjustment for polar shift
    def adjust_arctic_latitude(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Adjust latitude for Arctic basin calculations.

        Args:
            lat: Latitudes to be adjusted
            lon: Corresponding longitudes

        Returns:
            np.ndarray: Ajusted latitudes
        """
        # Points for Arctic basin 'West' of Lomonossov ridge
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
        """Load CANYON-B weights from assets folder"""
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

    def _predict(
        self,
        param: str,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
    ):
        """Private predictor to be used for a single parameter

        Parameters
        ----------

        param : str
            Parameters that will be predicted. Must be one of 'AT', 'DIC', 'pHT', 'pCO2', 'NO3', 'PO4', 'SiOH4'
        epres, etemp, epsal : float, optional
            Input errors
        edoxy : float or array-like, optional
            Oxygen input error (default: 1% of doxy)
        """
        # Get array shape and number of elements
        shape = self._obj[self._argo._TNAME].shape
        nol = self._argo.N_POINTS

        # Setting up default uncertainties if not provided
        epres = 0.5 if epres is None else epres
        etemp = 0.005 if etemp is None else etemp
        epsal = 0.005 if epsal is None else epsal
        edoxy = (
            0.01 * self._obj.DOXY.values if edoxy is None else edoxy
        )  # add case when DOXY_ADJUSTED is defined?

        # Expand scalar error values
        errors = [epres, etemp, epsal, edoxy]
        errors = [
            np.full(nol, e) if np.isscalar(e) else np.asarray(e).flatten()
            for e in errors
        ]
        epres, etemp, epsal, edoxy = errors

        # Define parameters and their properties
        paramnames = ["AT", "DIC", "pHT", "pCO2", "NO3", "PO4", "SiOH4"]
        i_idx = {p: i for i, p in enumerate(paramnames)}
        inputsigma = np.array([6, 4, 0.005, np.nan, 0.02, 0.02, 0.02])
        betaipCO2 = np.array([-3.114e-05, 1.087e-01, -7.899e01])

        # Adjust pH uncertainty (Orr systematic uncertainty)
        inputsigma[2] = np.sqrt(0.005**2 + 0.01**2)

        # Prepare input data
        data = self.create_canyonb_input_matrix()

        # Output dictionary
        out = {}

        #                                                         #
        # Process through neural networks for the given parameter #
        #                                                         #

        # Get index depending on parameter (mostly important to decipher between nutrients and carbonate sytems)
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

            # Forward pass
            a = np.dot(data_N, w1.T) + b1
            if nlayerflag == 1:
                # One hidden layer
                y = np.dot(np.tanh(a), w2.T) + b2
            else:
                # Two hidden layers
                b = np.dot(np.tanh(a), w2.T) + b2
                y = np.dot(np.tanh(b), w3.T) + b3

            # Store results
            cval[:, network] = y.flatten()
            cvalcy[network] = 1 / beta  # 'noise' variance

            # Calculate input effects
            x1 = w1[None, :, :] * (1 - np.tanh(a)[:, :, None] ** 2)

            if nlayerflag == 1:
                # One hidden layer
                inx = np.einsum("ij,...jk->...ik", w2, x1)[:, 0, :]
            else:
                # Two hidden layers
                x2 = w2[None, :, :] * (1 - np.tanh(b)[:, :, None] ** 2)
                inx = np.einsum("ij,...jk,...kl->...il", w3, x2, x1)[:, 0, :]
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
                out[param + unc] = outcalc["d_pCO2__d_par2"] * out[param + unc]

            out[param + "_cim"] = outcalc["d_pCO2__d_par2"] * np.reshape(
                out[param + "_cim"], shape
            )

        return out

    def predict(
        self,
        params: Union[str, List[str]] = None,
        epres: Optional[float] = None,
        etemp: Optional[float] = None,
        epsal: Optional[float] = None,
        edoxy: Optional[Union[float, np.ndarray]] = None,
    ) -> xr.Dataset:
        """
        Make predictions using the CANYON-B method.

        Parameters
        ----------
        params: str, List[str], optional, default=None
            List of parameters to predict. If None is specified, all possible parameters will be predicted.
        epres, etemp, epsal : float, optional
            Input errors
        edoxy : float or array-like, optional
            Oxygen input error (default: 1% of doxy)

        Returns
        -------
        :class:`xr.Dataset`
        """

        # Validation of requested parameters to predict:
        if params is None:
            params = self.output_list
        else:
            params = to_list(params)
        for p in params:
            if p not in self.output_list:
                raise ValueError(
                    "Invalid parameter ('%s') to predict, must be in [%s]"
                    % (p, ",".join(self.output_list))
                )

        # Make predictions of each of the requested parameters
        for param in params:
            out = self._predict(
                param, epres=epres, etemp=etemp, epsal=epsal, edoxy=edoxy
            )

            # Add predicted parameter to xr.Dataset
            self._obj[param] = xr.zeros_like(self._obj["TEMP"])
            self._obj[param].attrs = self.get_param_attrs(param)
            self._obj[param].values = out[param].astype(np.float32).squeeze()

            # CI
            self._obj[f"{param}_CI"] = xr.zeros_like(self._obj[param])
            self._obj[f"{param}_CI"].attrs = self.get_param_attrs(param)
            self._obj[f"{param}_CI"].attrs[
                "long_name"
            ] = f"Uncertainty on {self.get_param_attrs(param)['long_name']}"
            self._obj[f"{param}_CI"].values = (
                out[f"{param}_ci"].astype(np.float32).squeeze()
            )

            # CIM
            cim_value = out[f"{param}_cim"]
            if np.isscalar(cim_value):
                self._obj[f"{param}_CIM"] = xr.full_like(
                    self._obj[param], fill_value=float(cim_value), dtype=np.float32
                )
            else:
                self._obj[f"{param}_CIM"] = xr.zeros_like(self._obj[param])
                self._obj[f"{param}_CIM"].values = cim_value.astype(
                    np.float32
                ).squeeze()
            self._obj[f"{param}_CIM"].attrs = self.get_param_attrs(param)
            self._obj[f"{param}_CIM"].attrs[
                "long_name"
            ] = f"Measurement uncertainty on {self.get_param_attrs(param)['long_name']}"

            # CIN
            self._obj[f"{param}_CIN"] = xr.zeros_like(self._obj[param])
            self._obj[f"{param}_CIN"].attrs = self.get_param_attrs(param)
            self._obj[f"{param}_CIN"].attrs[
                "long_name"
            ] = f"Uncertainty for Bayesian neural network mapping on {self.get_param_attrs(param)['long_name']}"
            self._obj[f"{param}_CIN"].values = (
                out[f"{param}_cin"].astype(np.float32).squeeze()
            )

            # CII
            self._obj[f"{param}_CII"] = xr.zeros_like(self._obj[param])
            self._obj[f"{param}_CII"].attrs = self.get_param_attrs(param)
            self._obj[f"{param}_CII"].attrs[
                "long_name"
            ] = f"Uncertainty due to input errors on {self.get_param_attrs(param)['long_name']}"
            self._obj[f"{param}_CII"].values = (
                out[f"{param}_cii"].astype(np.float32).squeeze()
            )

        # Return xr.Dataset with predicted variables:
        if self._argo:
            self._argo.add_history(
                "Added CANYON-B predictions for [%s]" % (",".join(params))
            )

        return self._obj

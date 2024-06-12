from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, List

from ..utils import path2assets, to_list, cast_Argo_variable_type
from ..extensions import register_argodataset_accessor


@register_argodataset_accessor('canyon_med')
class CanyonMED:

    ne = 7
    """Number of inputs"""

    output_list = ['PO4', 'NO3', 'CT', 'SiOH4', 'PAT', 'pHT']
    """List of parameters that can be predicted with this Neural Network"""

    def __init__(self, obj):
        if isinstance(obj, xr.Dataset):
            self._obj = obj  # Xarray object
        else:
            self._obj = obj._obj  # Xarray object from ArgoAccessor
        self.n_list = 5
        self.path2coef = Path(path2assets).joinpath('canyon-med')
        self._input = None  # Private NN input dataframe

    def get_param_attrs(self, param: str) -> dict:
        """Provides attributes to be added to a given predicted parameter"""
        attrs = {}
        if param in ['PO4', 'NO3', 'CT', 'SiOH4', 'PAT']:
            attrs.update({'units': 'micromole/kg'})
        if param == 'pHT':
            attrs.update({'units': 'Total scale at insitu PTS'})

        if param == 'PO4':
            attrs.update({'long_name': 'Phosphate concentration'})

        if param == 'NO3':
            attrs.update({'long_name': 'Nitrate concentration'})

        if param == 'CT':
            attrs.update({'long_name': 'Total dissolved inorganic carbon'})

        if param == 'SiOH4':
            attrs.update({'long_name': 'Silicate concentration'})

        if param == 'PAT':
            attrs.update({'long_name': 'Total alkalinity'})

        if param == 'pHT':
            attrs.update({'long_name': 'Total pH'})

        attrs.update({'comment': 'Synthetic variable predicted using CANYON-MED'})
        attrs.update({'reference': 'http://dx.doi/10.3389/fmars.2020.00620'})

        return attrs

    def param2suff(self, param):
        """File suffix to use for a given parameter to predict"""
        if param == 'PO4':
            suff = 'phos'
        elif param == 'NO3':
            suff = 'nit'
        elif param == 'CT':
            suff = 'CT'
        elif param == 'SiOH4':
            suff = 'sil'
        elif param == 'PAT':
            suff = 'AT'
        elif param == 'pHT':
            suff = 'ph'
        return suff

    def load_normalisation_factors(self, param, subset='F'):
        suff = self.param2suff(param)

        moy_sub = pd.read_table(self.path2coef.joinpath("moy_%s_%s.txt" % (suff, subset)),
                              sep=" {3}",
                              header=None,
                              engine='python').values
        std_sub = pd.read_table(self.path2coef.joinpath("std_%s_%s.txt" % (suff, subset)),
                              sep=" {3}",
                              header=None,
                              engine='python').values
        return moy_sub, std_sub

    def load_weights(self, param, subset, i):
        suff = self.param2suff(param)

        b1 = pd.read_csv(self.path2coef.joinpath("poids_%s_b1_%s_%i.txt" % (suff, subset, i)), header=None)
        b2 = pd.read_csv(self.path2coef.joinpath("poids_%s_b2_%s_%i.txt" % (suff, subset, i)), header=None)
        b3 = pd.read_csv(self.path2coef.joinpath("poids_%s_b3_%s_%i.txt" % (suff, subset, i)), header=None)
        IW = pd.read_csv(self.path2coef.joinpath("poids_%s_IW_%s_%i.txt" % (suff, subset, i)), sep="\s+", header=None)
        LW1 = pd.read_csv(self.path2coef.joinpath("poids_%s_LW1_%s_%i.txt" % (suff, subset, i)), sep="\s+", header=None)
        LW2 = pd.read_csv(self.path2coef.joinpath("poids_%s_LW2_%s_%i.txt" % (suff, subset, i)), sep="\s+", header=None)

        # Using float128 arrays avoid the error or warning "overflow encountered in exp" raised by the
        # activation function
        b1 = np.array(b1, dtype=np.float128)
        b2 = np.array(b2, dtype=np.float128)
        b3 = np.array(b3, dtype=np.float128)

        return b1, b2, b3, IW, LW1, LW2

    @property
    def decimal_year(self):
        return self._obj['TIME'].dt.year + (86400 * self._obj['TIME'].dt.dayofyear
                                                + 3600 * self._obj['TIME'].dt.hour
                                                + self._obj['TIME'].dt.second) / (365.0 * 24 * 60 * 60)

    def ds2df(self) -> pd.DataFrame:
        """Create a NN input :class:`pd.DataFrame` from :class:`xr.Dataset`"""

        df = pd.DataFrame({'lat': self._obj['LATITUDE'],
                           'lon': self._obj['LONGITUDE'],
                           'dec_year': self.decimal_year,
                           'pres': self._obj['PRES'],
                           'temp': self._obj['TEMP'],
                           'psal': self._obj['PSAL'],
                           'doxy': self._obj['DOXY']
                           })

        # Modify pressure
        # > The pressure input is transformed according to the combination of a linear
        # and a logistic curve to limit the degrees of freedom of the ANN in deep
        # waters and to account for the large range of pressure values (from the
        # surface to 4000 m depth) and a non-homogeneous distribution of data
        # within this range
        # See Eq. 3 in 10.3389/fmars.2020.00620
        df['pres'] = df['pres'].apply(lambda x: (x / 2e4) + (1 / ((1 + np.exp(-x / 300)) ** 3)))

        return df

    @property
    def input(self) -> pd.DataFrame:
        """NN input :class:`pd.DataFrame`

        This :class:`pd.DataFrame` is stored internally to avoid to re-compute it for each prediction

        Returns
        -------
        :class:`pd.DataFrame`
        """
        # if self._input is None:
        #     self._input = self.ds2df()
        #
        # return self._input
        return self.ds2df()

    def _predict(self, param: str):
        """Private predictor to be used for a single parameter"""

        # Define the activation function between the neurons,
        # See Eq. 1 in 10.3389/fmars.2020.00620
        def custom_MF(x: np.ndarray):
            tmp = 1.7159 * ((np.exp((4 / 3) * x) - 1) / (np.exp((4 / 3) * x) + 1))
            return (tmp)

        # Read/create NN input as a dataframe:
        df = self.input

        # NORMALISATION OF THE PARAMETERS
        # See Eq. 2 in 10.3389/fmars.2020.00620
        # (The factor 2/3 brings at least 80% of the data in the range [−1;1])
        data_N = df.iloc[:, :self.ne].copy()
        moy_F, std_F = self.load_normalisation_factors(param, 'F')
        for i in range(self.ne):
            data_N.iloc[:, i] = (2 / 3) * ((df.iloc[:, i] - moy_F[:, i]) / std_F[:, i])
        data_N = np.array(data_N)

        # rx = data_N.shape[0]  # Not used
        param_outputs_s = []

        for i in range(1, self.n_list + 1):
            b1, b2, b3, IW, LW1, LW2 = self.load_weights(param, 'F', i)

            # Calculate a
            a = custom_MF(np.dot(data_N, IW.T).T + b1)

            # Calculate b
            b = custom_MF(np.dot(LW1, a) + b2)

            # Calculate y
            y = np.dot(LW2, b) + b3
            y = y.T

            # Calculate phos_outputs
            param_outputs = 1.5 * y * std_F[0][self.ne] + moy_F[0][self.ne]
            param_outputs_s.append(param_outputs)

        # reshape array
        param_outputs_s1 = np.array(param_outputs_s).T

        # load normalisation factors
        moy_G, std_G = self.load_normalisation_factors(param, 'G')

        # NORMALISATION OF THE PARAMETERS
        data_N = df.iloc[:, :self.ne].copy()
        for i in range(self.ne):
            data_N.iloc[:, i] = (2 / 3) * ((df.iloc[:, i] - moy_G[:, i]) / std_G[:, i])
        data_N = np.array(data_N)

        param_outputs_s = []
        for i in range(1, self.n_list + 1):
            b1, b2, b3, IW, LW1, LW2 = self.load_weights(param, 'G', i)

            # Calculate a
            a = custom_MF(np.dot(data_N, IW.T).T + b1)

            # Calculate b
            b = custom_MF(np.dot(LW1, a) + b2)

            # Calculate y
            y = np.dot(LW2, b) + b3
            y = y.T

            # Calculate phos_outputs
            param_outputs = 1.5 * y * std_G[0][self.ne] + moy_G[0][self.ne]
            param_outputs_s.append(param_outputs)

        # flatten array because it's nice #2
        param_outputs_s2 = np.array(param_outputs_s).T

        # concat F and G data
        param_outputs_s = np.hstack((np.squeeze(param_outputs_s1, axis=0), np.squeeze(param_outputs_s2, axis=0)))

        # neural network
        mean_nn = np.mean(param_outputs_s, axis=1)
        std_nn = np.std(param_outputs_s, axis=1, ddof=1)

        #
        return mean_nn, std_nn

    def predict(self, params: Union[str, List[str]] = None) -> xr.Dataset:
        """Make predictions using the CANYON-MED neural network

        Parameters
        ----------
        params: str, List[str], optional, default=None
            List of parameters to predict. If None is specified, all possible parameters will be predicted.

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
                raise ValueError("Invalid parameter ('%s') to predict, must be in [%s]" % (p, ",".join(self.output_list)))

        # Make predictions of each of the requested parameters:
        for param in params:
            mean_nn, std_nn = self._predict(param)

            self._obj[param] = xr.zeros_like(self._obj['TEMP'])
            self._obj[param].attrs = self.get_param_attrs(param)
            self._obj[param].values = mean_nn.astype(np.float32)

            self._obj['%s_ERROR' % param] = xr.zeros_like(self._obj[param])
            self._obj['%s_ERROR' % param].attrs = self.get_param_attrs(param)
            self._obj['%s_ERROR' % param].attrs['long_name'] = "Error on %s" % self.get_param_attrs(param)['long_name']
            self._obj['%s_ERROR' % param].values = std_nn.astype(np.float32)

        # Return xr.Dataset with predicted variables:
        return self._obj

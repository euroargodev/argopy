import xarray as xr
import pandas as pd
from typing import List, Union
from pathlib import Path
import json
import logging

from ..options import OPTIONS
from ..utils import check_wmo, to_list, path2assets
from ..stores import httpstore, filestore


log = logging.getLogger("argopy.related.coriolis_api")


class CoriolisAPI:
    """Interface to the Coriolis big data API

    The Coriolis Big Data API is a scalable, RESTful service designed by Ifremer to expose high‑volume, in‑situ oceanographic and environmental observations. It provides:

    - Fast, filtered queries on time‑series and vertical profiles (temperature, salinity, currents)
    - Geo‑temporal slicing in JSON for seamless integration in analysis workflows
    - Interactive exploration via Swagger UI and OpenAPI specification
    - FAIR‑compliant metadata (JSON‑LD, CodeMeta, CFF, RO‑Crate) to support reproducibility

    Whether you’re a researcher, data scientist, or operational user, this API delivers the data you need—on demand, with full provenance and traceability.


    Related:
    https://github.com/euroargodev/argopy/issues/47

    Doc:
    https://api-coriolis.ifremer.fr/swagger-ui/index.html

    Ifremer Doc:
    https://dev-ops.gitlab-pages.ifremer.fr/documentation/service_datasheet/scientific/oceanography/coriolis/dataAccessAPI/
    https://gitlab.ifremer.fr/coriolis/developpement/bigdata/co054702/coriolis_api

    """

    @property
    def version(self):
        """API version"""
        return "1.1"

    @property
    def api(self):
        """API url"""
        return "https://api-coriolis.ifremer.fr/v1.1"

    @property
    def valid_parameters(self):
        """List of valid parameter names to be retrieved by default"""
        return ["PRES", "PRES_CORE", "TEMP", "PSAL"]

    @property
    def valid_endpoints(self):
        """List of supported API valid endpoints"""
        return ["profiles_timelines", "profiles", "profiles_parameters", "platform", "profiles_by_id"]

    def __init__(self, **kwargs):
        self.fs = httpstore(cache=kwargs.get('cache', True), cachedir=kwargs.get('cachedir', OPTIONS['cachedir']))
        """Internal http-based file store"""

        self._assets_codes = filestore().open_json(
            Path(path2assets).joinpath("api_coriolis_parameter_codes.json")
        )
        self._param2code = None

    @property
    def code2param(self) -> dict:
        """Code number to parameter name dictionary"""
        p2c = self.param2code.copy()
        c2p = {}
        for param, code in p2c.items():
            c2p.update({code: param})
        return c2p

    @property
    def param2code(self) -> dict:
        """Parameter name to code number dictionary"""
        if self._param2code is None:
            d = {}
            for param in self.valid_parameters:
                for ext in ["", "_ADJUSTED", "_ADJUSTED_ERROR"]:
                    parameter = f"{param}{ext}"
                    if parameter in self._assets_codes["data"]["params"]:
                        code = self._assets_codes["data"]["params"][parameter]["code"]
                        d.update({parameter: int(code)})
            self._param2code = d
        return self._param2code

    def profiles_timelines(self, wmo: Union[int, List]) -> List:
        """Get timeline of profiles

        Raw data is a list of dict with:
        {'timestamp': Date (Epoch unix timestamp), 'measures': Number of measures}

        Examples
        --------
        .. code-block:: python
            :caption: Profiles timelines

            from argopy.related import CoriolisAPI

            wmo = 6902915
            # wmo = [6902915, 4903855]

            profiles_timelines = CoriolisAPI().profiles_timelines(wmo)
            profiles_timelines[0]

        Note
        ----
        This method return unix timestamp typed as :class:`pd.Timestamp`
        """
        wmo = check_wmo(wmo)
        uri = [f"{self.api}/profiles_timelines?platform={w}" for w in wmo]

        def castvariables(response):
            for irow, profile in enumerate(response):
                response[irow].update(
                    {"timestamp": pd.to_datetime(int(profile["timestamp"]), unit="ms")}
                )
            return response

        data = self.fs.open_mfjson(uri, preprocess=castvariables, errors="ignore")
        if len(wmo) == 1:
            return data[0]
        else:
            return data

    def profiles(
            self,
            wmo: int,
            start: pd.Timestamp,
            end: pd.Timestamp = None,
            parameter_code: Union[int, List] = 35,
    ) -> pd.DataFrame:
        """Retrieve one or more parameter data for one or more profile

        Returns
        -------
        :class:`pd.DataFrame`
        """
        data_type = "PF"  # Our use case for Argo profiles
        end = start + pd.Timedelta("1 days") if end is None else end
        ts_start, ts_end = int(start.timestamp()), int(end.timestamp())

        parameter_code = to_list(parameter_code)
        uri = [
            f"{self.api}/profiles?start={ts_start}&end={ts_end}&data_type={data_type}&parameter={code}&platform={wmo}"
            for code in parameter_code]

        def castvariables(response):
            for iprof, _ in enumerate(response["result"]):
                response["result"][iprof]["data"] = pd.DataFrame(
                    response["result"][iprof]["data"],
                    columns=["value", "zvalue", "valueqc", "zlevel", "zqc"],
                )
                response["result"][iprof]["parameter"] = int(
                    response["result"][iprof]["parameter"]
                )
            return response

        # Fetch data
        data = self.fs.open_mfjson(uri, preprocess=castvariables)
        if len(parameter_code) > 1:
            return data
        else:
            return data[0]

    def profiles_parameters(self, observation_id: Union[int, List] = None):
        """Retrieve parameter codes for one or more profile ID"""
        observation_id = to_list(observation_id)
        uri = [
            f"{self.api}/profiles_parameters?observation_id={obs}"
            for obs in observation_id
        ]

        data = self.fs.open_mfjson(uri)
        if len(observation_id) == 1:
            return data[0]
        else:
            return data

    def platform(self, wmo: int = None) -> pd.DataFrame:
        """Retrieve one float metadata

        Examples
        --------
        .. code-block:: python
            :caption: Float metadata

            from argopy.related import CoriolisAPI

            wmo = 6902915
            metadata = CoriolisAPI().platform(wmo)
            print(len(metadata), metadata.keys(), metadata['tech_platforms_array'])

            wmo = [6902915, 4903855]
            metadata = CoriolisAPI().platform(wmo)
            print(len(metadata), metadata.keys(), metadata[wmo[0]]['tech_platforms_array'])

        Note
        ----
        Technical data are returned as a :class:`pd.DataFrame`
        """
        wmo = check_wmo(wmo)
        uri = [f"{self.api}/platform?code={w}" for w in wmo]

        def castvariables(response):
            response["tech_platforms_array"] = pd.DataFrame(
                response["tech_platforms_array"]
            )
            response["platform_code"] = check_wmo(response["platform_code"])[0]
            return response

        data = self.fs.open_mfjson(uri, preprocess=castvariables)
        if len(wmo) == 1:
            response = data[0]
        else:
            response = {}
            for record in data:
                response.update({record["platform_code"]: record})
        return response

    def observations_id(self,
                        wmo: int,
            start: pd.Timestamp,):
        """Retrieve one observation ID"""
        return self.profiles(wmo, start)['result'][0]['observationId']

    def profiles_by_id(self, observation_id: Union[int, List] = None):
        observation_id = to_list(observation_id)
        uri = [f"{self.api}/profiles_by_id?id={obs}" for obs in observation_id]

        def castvariables(response):
            for iprof, _ in enumerate(response["result"]):
                response["result"][iprof]["data"] = pd.DataFrame(
                    response["result"][iprof]["data"],
                    columns=["value", "zvalue", "valueqc", "zlevel", "zqc"],
                )
                response["result"][iprof]["parameter"] = int(
                    response["result"][iprof]["parameter"]
                )
            return response

        data = self.fs.open_mfjson(uri, preprocess=castvariables)

        if len(uri) == 1:
            response = data[0]
        else:
            response = data
            # response = {}
            # for record in data:
            #     response.update({record["platform_code"]: record})
        return response

    def to_dataframe(
            self,
            wmo: int,
            start: pd.Timestamp,
    ):
        """Retrieve all possible parameter data for one profile as a :class:`pandas.DataFrame`"""

        # Retrieve TEMP parameter, to get the observation id
        profile = self.profiles(wmo, start, parameter_code=35)  # TEMP is always there

        # Get the list of parameters for this observation id:
        observationId = profile['result'][0]['observationId']
        codes = self.profiles_parameters(observationId)

        # Retrieve all parameters but 35/TEMP:
        my_codes = codes.copy()
        my_codes.remove(35)
        data = self.profiles(wmo, start, parameter_code=codes)

        # Reformat all dataframe with explicit column names:
        def rename(this_prof):
            df1 = this_prof['data']
            parameter = self.code2param[this_prof['parameter']]
            this_prof['data'] = df1.rename(
                {'value': parameter, 'valueqc': f"{parameter}_QC", 'zvalue': 'PRES', 'zqc': 'PRES_QC'}, axis=1)
            return this_prof

        profile['result'][0] = rename(profile['result'][0])
        for ic in range(len(codes)):
            data[ic]['result'][0] = rename(data[ic]['result'][0])

        # Align and merge all dataframes:
        def align_and_merge(df1, df2):
            # Columns from df2, already in df1 to be removed (but we keep zlevel for alignement):
            l = [key for key in df2.columns if key in df1.columns]
            l.remove('zlevel')
            return df1.merge(df2.copy().drop(l, axis=1), how='outer', on='zlevel')

        df = profile['result'][0]['data']
        for ic in range(len(codes)):
            df = align_and_merge(df, data[ic]['result'][0]['data'])
        df = df[sorted(df.columns)].reset_index(drop=True)

        return df

    def to_xarray(
            self,
            wmo: int,
            start: pd.Timestamp
    ):
        return xr.Dataset(self.to_dataframe(wmo, start))
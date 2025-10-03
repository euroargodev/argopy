from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Union
import logging

# from ..errors import InvalidOption
from ..stores import ArgoFloat, ArgoIndex, httpstore, filestore
from ..utils import check_wmo, Chunker, to_list, NVSrow
from ..errors import (
    DataNotFound,
    InvalidDataset,
    InvalidDatasetStructure,
    OptionValueError,
)
from ..options import OPTIONS
from ..utils import path2assets
from . import ArgoNVSReferenceTables


log = logging.getLogger("argopy.related.sensors")


class SensorType(NVSrow):
    """One single sensor type data from a R25-"Argo sensor types" row

    Examples
    --------
    .. code-block:: python
        :caption: Search methods

        from argopy import ArgoNVSReferenceTables

        df = ArgoNVSReferenceTables().tbl(25)

        sensor_type = 'CTD'
        st = SensorType.from_series(df[df["altLabel"].apply(lambda x: x == sensor_type)].iloc[0])

        st.name
        st.long_name
        st.definition
        st.deprecated
        st.uri

    """

    reftable = "R25"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorType":
        return SensorType(obj)


class SensorModel(NVSrow):
    """One single sensor model data from a R27-"Argo sensor models" row

    Examples
    --------
    .. code-block:: python
        :caption: Search methods

        from argopy import ArgoNVSReferenceTables

        df = ArgoNVSReferenceTables().tbl(27)

        sensor_model = 'AANDERAA_OPTODE_4330F'
        sm = SensorModel.from_series(df[df["altLabel"].apply(lambda x: x == sensor_model)].iloc[0])

        sm.name
        sm.long_name
        sm.definition
        sm.deprecated
        sm.uri
    """

    reftable = "R27"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorModel":
        return SensorModel(obj)


class ArgoSensor:
    def __init__(
        self,
        model: str = None,
        **kwargs,
    ):
        """Argo sensors helper class

        Parameters
        ----------
        model: str, optional
            An exact sensor model name, by default None.

            Allowed values can be obtained with:
            ``ArgoSensor().reference_model_name``

        Other Parameters
        ----------------
        cache : bool, optional, default: False
            Use cache or not for fetched data
        cachedir: str, optional, default: OPTIONS['cachedir']
            Folder where to store cached files.
        timeout: int, optional, default: OPTIONS['api_timeout']
            Time out in seconds to connect to web API

        Examples
        --------
        .. code-block:: python
            :caption: Access and search reference tables 25 and 27

            from argopy import ArgoSensor

            # Return reference table R27-"Argo sensor models" with the list of sensor models
            ArgoSensor().reference_model
            ArgoSensor().reference_model_name  # Only the list of names (used to fill 'SENSOR_MODEL' parameter)

            # Return reference table R25-"Argo sensor types" with the list of sensor types
            ArgoSensor().reference_sensor
            ArgoSensor().reference_sensor_type # Only the list of types (used to fill 'SENSOR' parameter)

            # Search for all (R27) referenced sensor models with some string in their name
            ArgoSensor().search_model('RBR')
            ArgoSensor().search_model('RBR', output='name') # Return a list of names instead of a DataFrame
            ArgoSensor().search_model('SBE41CP', strict=False)
            ArgoSensor().search_model('SBE41CP', strict=True)  # Exact string match required

        .. code-block:: python
            :caption: Search the Argo dataset for some sensor models and retrieve a list of WMOs or sensor serial numbers, or both

            from argopy import ArgoSensor

            # Search for sensor model name(s) having some string and return a list of WMOs equipped with it/them
            ArgoSensor().search('RBR', output='wmo')

            # Search for sensor model name(s) having some string and return a list of sensor serial numbers in Argo
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn')
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn', progress=True)

            # Search for sensor model name(s) having some string and return a list of tuples with WMOs and serial numbers for those equipped with this model
            ArgoSensor().search('SBE', output='wmo_sn')
            ArgoSensor().search('SBE', output='wmo_sn', progress=True)

        .. code-block:: python
            :caption: Search the Argo dataset for some sensor models and easily loop through `ArgoFloat` instances for each floats

            from argopy import ArgoSensor

            model = "RAFOS"
            for af in ArgoSensor().iterfloats_with(model):
                print(af.WMO)

            # Example for how to use the metadata attribute of an ArgoFloat instance:
            model = "RAFOS"
            for af in ArgoSensor().iterfloats_with(model):
                models = af.metadata['sensors']
                for s in models:
                    if model in s['model']:
                        print(af.WMO, s['maker'], s['model'], s['serial'])

        .. code-block:: python
            :caption: Use an exact sensor model name to create an instance

            from argopy import ArgoSensor

            sensor = ArgoSensor('RBR_ARGO3_DEEP6K')

            sensor.model
            sensor.type

            sensor.search(output='wmo')
            sensor.search(output='sn')
            sensor.search(output='wmo_sn')

        Notes
        -----
        Related ADMT/AVTT issues:
            - https://github.com/OneArgo/ADMT/issues/112
            - https://github.com/OneArgo/ArgoVocabs/issues/156
            - https://github.com/OneArgo/ArgoVocabs/issues/157
        """
        self._cache = kwargs.get("cache", True)
        self._cachedir = kwargs.get("cachedir", OPTIONS["cachedir"])
        self.timeout = kwargs.get("timeout", OPTIONS["api_timeout"])
        self.fs = httpstore(cache=self._cache, cachedir=self._cachedir)

        self._r25 = None  # will be loaded when necessary
        self._r27 = None  # will be loaded when necessary
        self._load_mappers()  # Load r25 model to r27 type mapping dictionary

        if model is not None:
            try:
                df = self.search_model(model, strict=True)
            except DataNotFound:
                raise DataNotFound(
                    f"No sensor model named '{model}', as per ArgoSensor().reference_model_name values, based on Ref. Table 27."
                )

            if df.shape[0] == 1:
                self._model = SensorModel.from_series(df)
                self._type = self.model_to_type(self._model, errors="ignore")
            else:
                raise InvalidDatasetStructure(
                    f"Found multiple sensor models with '{model}'. Restrict your sensor model name to only one value in: {to_list(df['altLabel'].values)}"
                )

        else:
            self._model = None
            self._type = None

    def _load_mappers(self):
        """Load the NVS R25 to R27 key mappings"""
        df = []
        for p in (
            Path(path2assets).joinpath("nvs_R25_R27").glob("NVS_R25_R27_mappings_*.txt")
        ):
            df.append(
                filestore().read_csv(
                    p,
                    header=None,
                    names=["origin", "model", "?", "destination", "type", "??"],
                )
            )
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        self.r27_to_r25 = {}
        df.apply(
            lambda row: self.r27_to_r25.update(
                {row["model"].strip(): row["type"].strip()}
            ),
            axis=1,
        )

    def model_to_type(
        self,
        model: Union[str, SensorModel] = None,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> Optional[SensorType]:
        """Read a sensor type for a given sensor model"""
        model_name = model.name if isinstance(model, SensorModel) else model
        sensor_type = self.r27_to_r25.get(model_name, None)
        if sensor_type is not None:
            row = self.reference_sensor[
                self.reference_sensor["altLabel"].apply(lambda x: x == sensor_type)
            ].iloc[0]
            return SensorType.from_series(row)
        elif errors == "raise":
            raise DataNotFound(
                f"Can't determine the type of sensor model '{model_name}' (no matching key in self.r27_to_r25 mapper)"
            )
        return None

    @property
    def model(self) -> SensorModel:
        if isinstance(self._model, SensorModel):
            return self._model
        else:
            raise InvalidDataset(
                "The 'model' property is not available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def type(self) -> SensorType:
        if isinstance(self._type, SensorType):
            return self._type
        else:
            raise InvalidDataset(
                "The 'type' property is not available for an ArgoSensor instance not created with a specific sensor model"
            )

    def __repr__(self):
        if isinstance(self._model, SensorModel):
            summary = [f"<argosensor.{self.type.name}.{self.model.name}>"]
            summary.append(f"TYPE➤ {self.type.long_name}")
            summary.append(f"MODEL➤ {self.model.long_name}")
            if self.model.deprecated:
                summary.append("⛔ This model is deprecated !")
            else:
                summary.append("✅ This model is not deprecated.")
            summary.append(f"🔗 {self.model.uri}")
            summary.append(f"❝{self.model.definition}❞")
        else:
            summary = ["<argosensor>"]
            summary.append(
                "This instance was not created with a sensor model name, you still have access to the following:"
            )
            summary.append("👉 attributes: ")
            for attr in [
                "reference_model",
                "reference_model_name",
                "reference_sensor",
                "reference_sensor_type",
            ]:
                summary.append(f"  ╰┈➤ ArgoSensor().{attr}")

            summary.append("👉 methods: ")
            for meth in [
                "search_model",
                "search_model_name",
                "search",
                "iterfloats_with",
            ]:
                summary.append(f"  ╰┈➤ ArgoSensor().{meth}()")
        return "\n".join(summary)

    @property
    def reference_model(self) -> pd.DataFrame:
        """Return the official reference table for Argo sensor models

        Return the Argo Reference table R27 'SENSOR_MODEL':

        > Terms listing models of sensors mounted on Argo floats.
        """
        if self._r27 is None:
            self._r27 = ArgoNVSReferenceTables(
                cache=self._cache, cachedir=self._cachedir
            ).tbl("R27")
        return self._r27

    @property
    def reference_model_name(self) -> List[str]:
        """Return the official list of Argo sensor models

        Return a sorted list of strings with altLabel from Argo Reference table R27 'SENSOR_MODEL'.

        Notes
        -----
        Argo netCDF variable ``SENSOR_MODEL`` is populated with values from this list.
        """
        return sorted(to_list(self.reference_model["altLabel"].values))

    @property
    def reference_sensor(self) -> pd.DataFrame:
        """Return the official list of Argo sensor types

        Return the Argo Reference table R25 'SENSOR':

        > Terms describing sensor types mounted on Argo floats.
        """
        if self._r25 is None:
            self._r25 = ArgoNVSReferenceTables(
                cache=self._cache, cachedir=self._cachedir
            ).tbl("R25")
        return self._r25

    @property
    def reference_sensor_type(self) -> List[str]:
        """Return the official list of Argo sensor types

        Return a sorted list of strings with altLabel from Argo Reference table R25 'SENSOR'.

        Notes
        -----
        Argo netCDF variable ``SENSOR`` is populated with values from this list.
        """
        return sorted(to_list(self.reference_sensor["altLabel"].values))

    def search_model(
        self,
        model: str,
        strict: bool = False,
        output: Literal["table", "name"] = "table",
    ) -> pd.DataFrame:
        """Return references of Argo sensor models matching a string

        Look for occurrences in Argo Reference table R27 altLabel values and return a dataframe with matching row(s).
        """
        if strict:
            data = self.reference_model[
                self.reference_model["altLabel"].apply(lambda x: x == model)
            ]
        else:
            data = self.reference_model[
                self.reference_model["altLabel"].apply(lambda x: model in x)
            ]
        if data.shape[0] == 0:
            if strict:
                raise DataNotFound(
                    f"No sensor models matching '{model}'. You may try to search with strict=False."
                )
            else:
                raise DataNotFound(
                    f"No sensor model names with '{model}' string occurrence."
                )
        else:
            if output == "name":
                return sorted(to_list(data["altLabel"].values))
            else:
                return data

    def _search_wmo_with(self, model: str, errors="raise"):
        """Return the list of WMOs with a given sensor model

        Notes
        -----
        Based on a fleet-monitoring API request to `platformCodes/multi-lines-search` on `sensorModels` field.

        Documentation:

        https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/platform-code-controller/getPlatformCodesMultiLinesSearchUsingPOST

        """
        api_point = f"{OPTIONS['fleetmonitoring']}/platformCodes/multi-lines-search"
        payload = [
            {
                "nested": False,
                "path": "string",
                "searchValueType": "Text",
                "values": [model],
                "field": "sensorModels",
            }
        ]
        wmos = self.fs.post(api_point, json_data=payload)
        if wmos is None or len(wmos) == 0:
            if errors == "raise":
                raise DataNotFound(f"No floats matching sensor model name '{model}'")
            else:
                log.error(f"No floats matching sensor model name '{model}'")
        return check_wmo(wmos)

    def _floats_api(
        self,
        model: str,
        preprocess=None,
        preprocess_opts={},
        postprocess=None,
        postprocess_opts={},
        progress=False,
        errors="raise",
    ):
        """Search floats with a sensor model and then fetch and process JSON data returned from the fleet-monitoring API for each floats

        Notes
        -----
        Based on a POST request to the fleet-monitoring API requests to `/floats/{wmo}`.

        `Endpoint documentation <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/autonomous-float-controller/getFullFloatUsingGET>`_.
        """
        wmos = self._search_wmo_with(model)

        URI = []
        for wmo in wmos:
            URI.append(f"{OPTIONS['fleetmonitoring']}/floats/{wmo}")

        sns = self.fs.open_mfjson(
            URI,
            preprocess=preprocess,
            preprocess_opts=preprocess_opts,
            progress=progress,
            errors=errors,
        )

        return postprocess(sns, **postprocess_opts)

    def _search_sn_with(self, model: str, progress=False, errors="raise"):
        """Return serial number of sensor models with a given string in name"""

        def preprocess(jsdata, model_name: str = ""):
            sn = np.unique(
                [s["serial"] for s in jsdata["sensors"] if model_name in s["model"]]
            )
            return sn

        def postprocess(data, **kwargs):
            S = []
            for row in data:
                for sensor in row:
                    S.append(sensor)
            return np.sort(np.array(S))

        return self._floats_api(
            model,
            preprocess=preprocess,
            preprocess_opts={"model_name": model},
            postprocess=postprocess,
            progress=progress,
            errors=errors,
        )

    def _search_wmo_sn_with(self, model: str, progress=False, errors="raise"):
        """Return a dictionary of float WMOs with their sensor serial numbers"""

        def preprocess(jsdata, model_name: str = ""):
            sn = np.unique(
                [s["serial"] for s in jsdata["sensors"] if model_name in s["model"]]
            )
            return [jsdata["wmo"], [str(s) for s in sn]]

        def postprocess(data, **kwargs):
            S = {}
            for wmo, sn in data:
                S[check_wmo(wmo)[0]] = to_list(sn)
            return S

        return self._floats_api(
            model,
            preprocess=preprocess,
            preprocess_opts={"model_name": model},
            postprocess=postprocess,
            progress=progress,
            errors=errors,
        )

    def search(
        self,
        model: str = None,
        output: Literal["wmo", "sn", "wmo_sn"] = "wmo",
        progress=False,
        errors="raise",
    ):
        """Search for Argo floats equipped with a sensor model name

        All information are retrieved using the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu>`_.

        Parameters
        ----------
        model: str, optional
            A string to search in the `sensorModels` field of the Euro-Argo fleet-monitoring API `platformCodes/multi-lines-search` endpoint.

        output: str, Literal["wmo", "sn", "wmo_sn"], default "wmo"
            Define the output to return:

                - "wmo": a list of WMO numbers (integers)
                - "sn": a list of sensor serial numbers (strings)
                - "wmo_sn": a list of dictionary with WMO as key and serial numbers as values

        progress: bool, default False
            Define whether to display a progress bar or not

        errors: str, default "raise"

        Returns
        -------
        List[int], List[str], Dict

        Notes
        -----
        The list of WMOs equipped with a given sensor model is retrieved using the Euro-Argo fleet-monitoring API and a request to the `platformCodes/multi-lines-search` endpoint using the `sensorModels` search field.

        Sensor serial numbers are given by float meta-data retrieved using the Euro-Argo fleet-monitoring API and a request to the `/floats/{wmo}` endpoint:

        See Also
        --------
        `Endpoint 'platformCodes/multi-lines-search' documentation <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/platform-code-controller/getPlatformCodesMultiLinesSearchUsingPOST>`_.

        `Endpoint '/floats/{wmo}' documentation <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/autonomous-float-controller/getFullFloatUsingGET>`_.

        """
        if model is None and self.model is not None:
            model = self.model.name
        if output == "wmo":
            return self._search_wmo_with(model=model, errors=errors)
        elif output == "sn":
            return self._search_sn_with(model=model, progress=progress, errors=errors)
        elif output == "wmo_sn":
            return self._search_wmo_sn_with(
                model=model, progress=progress, errors=errors
            )
        else:
            raise OptionValueError(
                "'output' option value must be in: 'wmo', 'sn', or 'wmo_sn'"
            )

    def iterfloats_with(self, model: str = None, chunksize: int = None):
        """Iterator over :class:`ArgoFloat` equipped with a given sensor model

        By default, iterate over a single float, otherwise use the `chunksize` argument to iterate over chunk of floats.

        Parameters
        ----------
        model: str
            A string to search in the `sensorModels` field of the Euro-Argo fleet-monitoring API `platformCodes/multi-lines-search` endpoint.

        chunksize: int, optional
            Maximum chunk size

            Eg: A value of 5 will create chunks with as many as 5 WMOs each.

        Returns
        -------
        Iterator of :class:`ArgoFloat`

        Examples
        --------
        .. code-block:: python
            :caption: Example of iteration

            for af in ArgoSensor().iterfloats_with('SBE41CP'):
                af # is a ArgoFloat instance

        """
        if model is None and self.model is not None:
            model = self.model.name

        wmos = self._search_wmo_with(model=model)

        idx = ArgoIndex(
            index_file="core",
            cache=self._cache,
            cachedir=self._cachedir,
        )

        if chunksize is not None:
            chk_opts = {}
            chk_opts.update({"chunks": {"wmo": "auto"}})
            chk_opts.update({"chunksize": {"wmo": chunksize}})
            chunked = Chunker(
                {"wmo": self._search_wmo_with(model=model)}, **chk_opts
            ).fit_transform()
            for grp in chunked:
                yield [ArgoFloat(wmo, idx=idx) for wmo in grp]

        else:
            for wmo in wmos:
                yield ArgoFloat(wmo, idx=idx)

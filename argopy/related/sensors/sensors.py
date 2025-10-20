import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Any, Iterator
import logging

from ...stores import ArgoFloat, ArgoIndex, httpstore, filestore
from ...utils import check_wmo, Chunker, to_list, NVSrow
from ...errors import (
    DataNotFound,
    InvalidDataset,
    InvalidDatasetStructure,
    OptionValueError,
)
from ...options import OPTIONS
from ...utils import path2assets
from .. import ArgoNVSReferenceTables


SearchOutputOptions = Literal["wmo", "sn", "wmo_sn", "df"]
ErrorOptions = Literal["raise", "ignore"]

log = logging.getLogger("argopy.related.sensors")


class SensorType(NVSrow):
    """One single sensor type data from a R25-"Argo sensor types" row

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoNVSReferenceTables

        sensor_type = 'CTD'

        df = ArgoNVSReferenceTables().tbl(25)
        df_match = df[df["altLabel"].apply(lambda x: x == sensor_type)].iloc[0]

        st = SensorType.from_series(df_match)

        st.name
        st.long_name
        st.definition
        st.deprecated
        st.uri

    """

    reftable = "R25"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorType":
        """Create a :class:`SensorType` from a R25-"Argo sensor models" row"""
        return SensorType(obj)


class SensorModel(NVSrow):
    """One single sensor model data from a R27-"Argo sensor models" row

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoNVSReferenceTables

        sensor_model = 'AANDERAA_OPTODE_4330F'

        df = ArgoNVSReferenceTables().tbl(27)
        df_match = df[df["altLabel"].apply(lambda x: x == sensor_model)].iloc[0]

        sm = SensorModel.from_series(df_match)

        sm.name
        sm.long_name
        sm.definition
        sm.deprecated
        sm.uri
    """

    reftable = "R27"

    @staticmethod
    def from_series(obj: pd.Series) -> "SensorModel":
        """Create a :class:`SensorModel` from a R27-"Argo sensor models" row"""
        return SensorModel(obj)

    def __contains__(self, string) -> bool:
        return string.lower() in self.name.lower() or string.lower() in self.long_name.lower()

class ArgoSensor:
    """Argo sensor 'package' helper class

    A :class:`ArgoSensor` class instance shall represent one float sensor 'package'


    The :class:`ArgoSensor` class aims to provide direct access to Argo's sensor metadata from:

    - NVS Reference Tables `Sensor Types (R25) <http://vocab.nerc.ac.uk/collection/R25>`_ and `Sensor Models (R27) <http://vocab.nerc.ac.uk/collection/R27>`_
    - `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu/>`_

    This enables users to:

    - navigate reference tables 25 and 27,
    - search for/iterate over floats equipped with specific sensor models,
    - retrieve sensor serial numbers across the global array,

    """

    __slots__ = ["_cache", "_cachedir", "_timeout", "fs", "_r25", "_r26", "_r27", "_r27_to_r25", "_model", "_type"]

    def __init__(
        self,
        model: str | None = None,
        **kwargs,
    ):
        """Create an instance of :class:`ArgoSensor`

        Parameters
        ----------
        model: str, optional
            An exact sensor model name, by default set to None because this is optional.

            Allowed possible values can be obtained with:
            :class:`ArgoSensor.reference_model_name`

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
            :caption: Access and search reference tables

            from argopy import ArgoSensor

            # Reference table R27-"Argo sensor models" with the list of sensor models
            ArgoSensor().reference_model
            ArgoSensor().reference_model_name  # Only the list of names (used to fill 'SENSOR_MODEL' parameter)

            # Reference table R25-"Argo sensor types" with the list of sensor types
            ArgoSensor().reference_sensor
            ArgoSensor().reference_sensor_type # Only the list of types (used to fill 'SENSOR' parameter)

            # Reference table R26-"Argo sensor manufacturers" with the list of sensor maker
            ArgoSensor().reference_manufaturer
            ArgoSensor().reference_manufaturer_name # Only the list of makers (used to fill 'SENSOR_MAKER' parameter)

            # Search for all referenced sensor models with some string in their name
            ArgoSensor().search_model('RBR')
            ArgoSensor().search_model('RBR', output='name') # Return a list of names instead of a DataFrame
            ArgoSensor().search_model('SBE41CP', strict=False)
            ArgoSensor().search_model('SBE41CP', strict=True)  # Exact string match required

        .. code-block:: python
            :caption: Search for Argo floats with some sensor models

            from argopy import ArgoSensor

            # Search and return a list of WMOs equipped with it/them
            ArgoSensor().search('RBR', output='wmo')

            # Search and return a list of sensor serial numbers in Argo
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn')
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn', progress=True)

            # Search and return a list of tuples with WMOs and serial numbers for those equipped with this model
            ArgoSensor().search('SBE', output='wmo_sn')
            ArgoSensor().search('SBE', output='wmo_sn', progress=True)

            # Search and return a DataFrame with full sensor information from floats equipped
            ArgoSensor().search('RBR', output='df')

        .. code-block:: python
            :caption: Easily loop through `ArgoFloat` instances for each floats equipped with a sensor model

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

        .. code-block:: bash
            :caption: Get clean search results from the command-line with :class:`ArgoSensor.cli_search`

            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo')"

            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='sn')"


        Notes
        -----
        Related ADMT/AVTT work:
            - https://github.com/OneArgo/ADMT/issues/112
            - https://github.com/OneArgo/ArgoVocabs/issues/156
            - https://github.com/OneArgo/ArgoVocabs/issues/157
        """
        self._cache = kwargs.get("cache", True)
        self._cachedir = kwargs.get("cachedir", OPTIONS["cachedir"])
        self._timeout = kwargs.get("timeout", OPTIONS["api_timeout"])
        fs_kargs = {"cache": self._cache, "cachedir": self._cachedir, "timeout": self._timeout}
        self.fs = httpstore(**fs_kargs)

        self._r25: pd.DataFrame | None = None  # will be loaded when necessary
        self._r26: pd.DataFrame | None = None  # will be loaded when necessary
        self._r27: pd.DataFrame | None = None  # will be loaded when necessary
        self._load_mappers()  # Load r25 model to r27 type mapping dictionary

        self._model: SensorModel | None = None
        self._type: SensorType | None = None
        if model is not None:
            try:
                df = self.search_model(model, strict=True)
            except DataNotFound:
                raise DataNotFound(
                    f"No sensor model named '{model}', as per ArgoSensor().reference_model_name values, based on Ref. Table 27."
                )

            if df.shape[0] == 1:
                self._model = SensorModel.from_series(df.iloc[0])
                self._type = self.model_to_type(self._model, errors="ignore")
                # if "RBR" in self._model:
                    # Add the RBR OEM API Authorization key for this sensor:
                    # fs_kargs.update(client_kwargs={'headers': {'Authorization': OPTIONS.get('rbr_api_key') }})
            else:
                raise InvalidDatasetStructure(
                    f"Found multiple sensor models with '{model}'. Restrict your sensor model name to only one value in: {to_list(df['altLabel'].values)}"
                )


    def _load_mappers(self):
        """Load from static assets file the NVS R25 to R27 key mappings

        These mapping files were download from https://github.com/OneArgo/ArgoVocabs/issues/156.
        """
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
        self._r27_to_r25: dict[str, str] = {}
        df.apply(
            lambda row: self._r27_to_r25.update(
                {row["model"].strip(): row["type"].strip()}
            ),
            axis=1,
        )

    @property
    def r27_to_r25(self) -> dict[str, str]:
        """Dictionary mapping of R27 to R25

        This mapping is from files for group 1, 2, 2b, 3, 3b and 4(s) downloaded on 2025/10/03 from https://github.com/OneArgo/ArgoVocabs/issues/156.

        Returns
        -------
        dict[str, str]

        Notes
        -----
        If you think a key is missing or that mapping files bundled with argopy are out of date, please raise an issue at https://github.com/euroargodev/argopy/issues.
        """
        if self._r27_to_r25 is None:
            self._load_mappers()
        return self._r27_to_r25

    def model_to_type(
        self,
        model: str | SensorModel | None = None,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> SensorType | None:
        """Get a sensor type for a given sensor model

        All valid sensor model name can be obtained with :attr:`ArgoSensor.reference_model_name`.

        Mapping between sensor model name (R27) and sensor type (R25) are from AVTT work at https://github.com/OneArgo/ArgoVocabs/issues/156.

        Parameters
        ----------
        model : str | :class:`SensorModel`
            The model to read the sensor type for.
        errors : Literal["raise", "ignore"] = "raise"
            How to handle possible errors. If set to "ignore", the method will return None.

        Returns
        -------
        :class:`SensorType` | None

        See Also
        --------
        :attr:`ArgoSensor.type_to_model`
        """
        model_name: str = model.name if isinstance(model, SensorModel) else model
        sensor_type = self.r27_to_r25.get(model_name, None)
        if sensor_type is not None:
            row = self.reference_sensor[
                self.reference_sensor["altLabel"].apply(lambda x: x == sensor_type)
            ].iloc[0]
            return SensorType.from_series(row)
        elif errors == "raise":
            raise DataNotFound(
                f"Can't determine the type of sensor model '{model_name}' (no matching key in ArgoSensor().r27_to_r25 mapper)"
            )
        return None

    def type_to_model(
        self,
        type: str | SensorType,
        errors: Literal["raise", "ignore"] = "raise",
    ) -> list[str] | None:
        """Get all sensor model names of a given sensor type

        All valid sensor types can be obtained with :attr:`ArgoSensor.reference_sensor_type`

        Mapping between sensor model name (R27) and sensor type (R25) are from AVTT work at https://github.com/OneArgo/ArgoVocabs/issues/156.

        Parameters
        ----------
        type : str, :class:`SensorType`
            The sensor type to read the sensor model name for.
        errors : Literal["raise", "ignore"] = "raise"
            How to handle possible errors. If set to "ignore", the method will return None.

        Returns
        -------
        list[str]

        See Also
        --------
        :attr:`ArgoSensor.model_to_type`
        """
        sensor_type = type.name if isinstance(type, SensorType) else type
        result = []
        for key, val in self.r27_to_r25.items():
            if sensor_type.lower() in val.lower():
                row = self.reference_model[
                    self.reference_model["altLabel"].apply(lambda x: x == key)
                ].iloc[0]
                result.append(SensorModel.from_series(row).name)
        if len(result) == 0:
            if errors == "raise":
                raise DataNotFound(
                    f"Can't find any sensor model for this type '{sensor_type}' (no matching key in ArgoSensor().r27_to_r25 mapper)"
                )
            else:
                return None
        else:
            return result

    @property
    def model(self) -> SensorModel:
        """:class:`SensorModel` of this class instance

        Only available for a class instance created with an explicit sensor model name.

        Returns
        -------
        :class:`SensorModel`

        Raises
        ------
        :class:`InvalidDataset`
        """
        if isinstance(self._model, SensorModel):
            return self._model
        else:
            raise InvalidDataset(
                "The 'model' property is not available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def type(self) -> SensorType:
        """:class:`SensorType` of this class instance sensor model

        Only available for a class instance created with an explicit sensor model name.

        Returns:
        -------
        :class:`SensorType`

        Raises
        ------
        :class:`InvalidDataset`
        """
        if isinstance(self._type, SensorType):
            return self._type
        else:
            raise InvalidDataset(
                "The 'type' property is not available for an ArgoSensor instance not created with a specific sensor model"
            )

    def __repr__(self) -> str:
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
        """Official reference table for Argo sensor models (R27)

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgoNVSReferenceTables`
        """
        if self._r27 is None:
            self._r27 = ArgoNVSReferenceTables(fs=self.fs).tbl("R27")
        return self._r27

    @property
    def reference_model_name(self) -> list[str]:
        """Official list of Argo sensor models (R27)

        Return a sorted list of strings with altLabel from Argo Reference table R27 on 'SENSOR_MODEL'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR_MODEL`` is populated with values from this list.

        See Also
        --------
        :attr:`ArgoSensor.reference_model`
        """
        return sorted(to_list(self.reference_model["altLabel"].values))

    @property
    def reference_sensor(self) -> pd.DataFrame:
        """Official reference table for Argo sensor types (R25)

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgoNVSReferenceTables`
        """
        if self._r25 is None:
            self._r25 = ArgoNVSReferenceTables(fs=self.fs).tbl("R25")
        return self._r25

    @property
    def reference_sensor_type(self) -> list[str]:
        """Official list of Argo sensor types (R25)

        Return a sorted list of strings with altLabel from Argo Reference table R25 on 'SENSOR'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR`` is populated with values from this list.

        See Also
        --------
        :attr:`ArgoSensor.reference_sensor`
        """
        return sorted(to_list(self.reference_sensor["altLabel"].values))

    @property
    def reference_manufacturer(self) -> pd.DataFrame:
        """Official reference table for Argo sensor manufacturers (R26)

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgoNVSReferenceTables`
        """
        if self._r26 is None:
            self._r26 = ArgoNVSReferenceTables(fs=self.fs).tbl("R26")
        return self._r26

    @property
    def reference_manufacturer_name(self) -> list[str]:
        """Official list of Argo sensor maker (R26)

        Return a sorted list of strings with altLabel from Argo Reference table R26 on 'SENSOR_MAKER'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR_MAKER`` is populated with values from this list.

        See Also
        --------
        :attr:`ArgoSensor.reference_manufacturer`
        """
        return sorted(to_list(self.reference_manufacturer["altLabel"].values))

    def search_model(
        self,
        model: str,
        strict: bool = False,
        output: Literal["df", "name"] = "df",
    ) -> pd.DataFrame | list[str]:
        """Return references of Argo sensor models matching a string

        Look for occurrences in Argo Reference table R27 `altLabel` and return a :class:`pandas.DataFrame` with matching row(s).

        Parameters
        ----------
        model : str
            The model to search for.
        strict : bool, optional, default: False
            Is the model string a strict match or an occurrence in table look up.
        output : str, Literal["df", "name"], default "df"
            Is the output a :class:`pandas.DataFrame` with matching rows from :attr:`ArgoSensor.reference_model`, or a list of string.

        Returns
        -------
        :class:`pandas.DataFrame`, list[str]

        See Also
        --------
        :class:`ArgoSensor.reference_model`
        """
        if strict:
            data = self.reference_model[
                self.reference_model["altLabel"].apply(lambda x: x == model.upper())
            ]
        else:
            data = self.reference_model[
                self.reference_model["altLabel"].apply(lambda x: model.upper() in x)
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
                return data.reset_index(drop=True)

    def _search_wmo_with(self, model: str, errors : ErrorOptions = "raise") -> list[int]:
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
            try:
                search_hint: list[str] = self.search_model(
                    model, output="name", strict=False
                )
                msg = (
                    f"No floats matching this sensor model name '{model}'. Possible hint: %s"
                    % ("; ".join(search_hint))
                )
            except DataNotFound:
                msg = f"No floats matching this sensor model name '{model}'"
            if errors == "raise":
                raise DataNotFound(msg)
            else:
                log.error(msg)
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
    ) -> Any:
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

    def _search_sn_with(self, model: str, progress=False, errors : ErrorOptions = "raise") -> list[str]:
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
                    if sensor is not None:
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

    def _search_wmo_sn_with(
        self, model: str, progress=False, errors="raise"
    ) -> dict[int, str]:
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

    def _to_dataframe(self, model: str, progress=False, errors : ErrorOptions = "raise") -> pd.DataFrame:
        """Return a DataFrame with WMO, sensor type, model, maker, sn, units, accuracy and resolution

        Parameters
        ----------
        model: str, optional
            A string to search in the `sensorModels` field of the Euro-Argo fleet-monitoring API `platformCodes/multi-lines-search` endpoint.

        """
        if model is None and self.model is not None:
            model = self.model.name

        def preprocess(jsdata, model_name: str = ""):
            output = []
            for s in jsdata["sensors"]:
                if model_name in s["model"]:
                    this = [jsdata["wmo"]]
                    [
                        this.append(s[key])  # type: ignore
                        for key in [
                            "id",
                            "maker",
                            "model",
                            "serial",
                            "units",
                            "accuracy",
                            "resolution",
                        ]
                    ]
                    output.append(this)
            return output

        def postprocess(data, **kwargs):
            d = []
            for this in data:
                for wmo, sid, maker, model, sn, units, accuracy, resolution in this:
                    d.append(
                        {
                            "WMO": wmo,
                            "Type": sid,
                            "Model": model,
                            "Maker": maker,
                            "SerialNumber": sn,
                            "Units": units,
                            "Accuracy": accuracy,
                            "Resolution": resolution,
                        }
                    )
            return pd.DataFrame(d).sort_values(by="WMO").reset_index(drop=True)

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
        model: str | None = None,
        output: SearchOutputOptions = "wmo",
        progress : bool = False,
        errors : ErrorOptions = "raise",
    ) -> list[int] | list[str] | dict[int, str] | pd.DataFrame:
        """Search for Argo floats equipped with a sensor model name

        All information are retrieved using the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu>`_.

        Parameters
        ----------
        model: str, optional
            A string to search in the ``sensorModels`` field of the Euro-Argo fleet-monitoring API ``platformCodes/multi-lines-search`` endpoint.

        output: str, Literal["wmo", "sn", "wmo_sn", "df"], default "wmo"
            Define the output to return:

            - ``wmo``: a list of WMO numbers (integers)
            - ``sn``: a list of sensor serial numbers (strings)
            - ``wmo_sn``: a list of dictionary with WMO as key and serial numbers as values
            - ``df``: a :class:`pandas.DataFrame` with WMO, sensor type/model/maker and serial number

        progress: bool, default False
            Define whether to display a progress bar or not

        errors: str, default "raise"

        Returns
        -------
        list[int], list[str], dict[int, str], :class:`pandas.DataFrame`

        Notes
        -----
        The list of WMOs equipped with a given sensor model is retrieved using the Euro-Argo fleet-monitoring API and a request to the ``platformCodes/multi-lines-search`` endpoint using the ``sensorModels`` search field.

        Sensor serial numbers are given by float meta-data retrieved using the Euro-Argo fleet-monitoring API and a request to the ``/floats/{wmo}`` endpoint:

        - `Documentation for endpoint: platformCodes/multi-lines-search <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/platform-code-controller/getPlatformCodesMultiLinesSearchUsingPOST>`_.

        - `Documentation for endpoint: /floats/{wmo} <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/autonomous-float-controller/getFullFloatUsingGET>`_.

        """
        if model is None and self.model is not None:
            model = self.model.name
        if output == "df":
            return self._to_dataframe(model=model, progress=progress, errors=errors)
        elif output == "wmo":
            return self._search_wmo_with(model=model, errors=errors)
        elif output == "sn":
            return self._search_sn_with(model=model, progress=progress, errors=errors)
        elif output == "wmo_sn":
            return self._search_wmo_sn_with(
                model=model, progress=progress, errors=errors
            )
        else:
            raise OptionValueError(
                "'output' option value must be in: 'wmo', 'sn', 'wmo_sn' or 'df'"
            )

    def iterfloats_with(
        self, model: str | None = None, chunksize: int | None = None
    ) -> Iterator[ArgoFloat]:
        """Iterator over :class:`argopy.ArgoFloat` equipped with a given sensor model

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
        Iterator of :class:`argopy.ArgoFloat`

        Examples
        --------
        .. code-block:: python
            :caption: Example of iteration

            from argopy import ArgoSensor

            for afloat in ArgoSensor().iterfloats_with("SATLANTIC_PAR"):
                print(f"\n-Float {afloat.WMO}: Platform description = {afloat.metadata['platform']['description']}")
                for sensor in afloat.metadata["sensors"]:
                    if "SATLANTIC_PAR" in sensor["model"]:
                        print(f"  - Sensor Maker: {sensor['maker']}, Serial: {sensor['serial']}")

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

    def cli_search(self, model: str, output: SearchOutputOptions = "wmo") -> str:  # type: ignore
        """Quick sensor lookups from the terminal

        This function is a command-line-friendly output for float search (e.g., for piping to other tools).

        Examples
        --------
        .. code-block:: bash
            :caption: Example of search results from the command-line

            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo')"

            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='sn')"

        """
        results = self.search(model, output=output)
        print("\n".join(map(str, results)))

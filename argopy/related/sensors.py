from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, ClassVar, Literal,Union
import warnings
import logging

# from ..errors import InvalidOption
from ..stores import ArgoFloat, ArgoIndex, httpstore, filestore
from ..utils import check_wmo, Chunker, to_list
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


@dataclass
class NVSrow:
    """This class makes it easier to work with a single :class:`pd.DataFrame` row as an object"""
    name: str = ""
    long_name: str = ""
    definition: str = ""
    uri: str = ""
    deprecated: bool = None

    reftable : ClassVar[str]
    """Reference table"""

    def __init__(self, row: pd.Series):
        if not isinstance(row, pd.Series) and isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        row = row.to_dict()
        self.name = row['altLabel']
        self.long_name = row['prefLabel']
        self.definition = row['definition']
        self.deprecated = row['deprecated']
        self.uri = row['id']

    @staticmethod
    def from_series(obj: pd.Series) -> 'NVSrow':
        return NVSrow(obj)

    def __eq__(self, obj):
        return self.name == obj


class SensorType(NVSrow):
    """One single sensor type data from a R25 row"""
    reftable = 'R25'

    @staticmethod
    def from_series(obj: pd.Series) -> 'SensorType':
        return SensorType(obj)


class SensorModel(NVSrow):
    """One single sensor model data from a R27 row"""
    reftable = 'R27'

    @staticmethod
    def from_series(obj: pd.Series) -> 'SensorModel':
        return SensorModel(obj)


class ArgoSensor:
    def __init__(
        self,
        model: str = None,
        **kwargs,
    ):
        """Create an ArgoSensor helper class instance

        Parameters
        ----------
        model: str, optional
            A sensor model to use, by default None.

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
            :caption: ?

            from argopy import ArgoSensor

            # Return the reference table R27 with the list of sensor models
            ArgoSensor().reference_model
            ArgoSensor().reference_model_name  # Only the list of names (used to fill 'SENSOR_MODEL')

            # Return the reference table R25 with the list of sensor types
            ArgoSensor().reference_sensor
            ArgoSensor().reference_sensor_type # Only the list of types (used to fill 'SENSOR')

            # Return all (R27) referenced sensor models with some string in their name
            ArgoSensor().search_model('RBR')
            ArgoSensor().search_model('RBR', output='name') # Return list of names instead of dataframe
            ArgoSensor().search_model('SBE41CP', strict=False)
            ArgoSensor().search_model('SBE41CP', strict=True)

            # Return list of WMOs equipped with a sensor model name having some string
            ArgoSensor().search('RBR', output='wmo')

            # Return list of sensor serial number for a sensor model name having some string
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn')
            ArgoSensor().search('RBR_ARGO3_DEEP6K', output='sn', progress=True)

            # Return dict of WMOs with sensor serial number for a sensor model name having some string
            ArgoSensor().search('SBE', output='wmo_sn')
            ArgoSensor().search('SBE', output='wmo_sn', progress=True)

            # Loop through each (or chunks) of ArgoFloat instances for floats equipped with a sensor model name having some string
            for af in ArgoSensor().iterfloats_with("RAFOS"):
                print(af.WMO)

            # Same loop, show casing how to use the metadata attribute of an ArgoFloat instance:
            model = "RAFOS"
            for af in ArgoSensor().iterfloats_with(model):
                models = af.metadata['sensors']
                for s in models:
                    if model in s['model']:
                        print(af.WMO, s['maker'], s['model'], s['serial'])

        Notes
        -----
        Related NVS issues:
            - https://github.com/OneArgo/ADMT/issues/112
            - https://github.com/OneArgo/ArgoVocabs/issues/156
            - https://github.com/OneArgo/ArgoVocabs/issues/157
        """
        self._cache = kwargs.get('cache', True)
        self._cachedir = kwargs.get('cachedir', OPTIONS["cachedir"])
        self.timeout = kwargs.get('timeout', OPTIONS["api_timeout"])
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
                self._type = self.model_to_type(self._model, errors='ignore')
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
        for p in Path(path2assets).joinpath("nvs_R25_R27").glob("NVS_R25_R27_mappings_*.txt"):
            df.append(filestore().read_csv(p, header=None, names=["origin", "model", "?", "destination", "type", "??"]))
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        self.r27_to_r25 = {}
        df.apply(lambda row: self.r27_to_r25.update({row['model'].strip(): row['type'].strip()}), axis=1)

    def model_to_type(self, model: Union[str, SensorModel] = None, errors : Literal['raise', 'ignore'] = 'raise') -> Optional[SensorType]:
        """Read a sensor type for a given sensor model"""
        model_name = model.name if isinstance(model, SensorModel) else model
        sensor_type = self.r27_to_r25.get(model_name, None)
        if sensor_type is not None:
            row = self.reference_sensor[
                self.reference_sensor["altLabel"].apply(lambda x: x == sensor_type)
            ].iloc[0]
            return SensorType.from_series(row)
        elif errors == 'raise':
            raise DataNotFound(f"Can't determine the type of sensor model '{model_name}' (no matching key in self.r27_to_r25 mapper)")
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
                "search_wmo_with",
                "search_sn_with",
                "search_wmo_sn_with",
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

    def search_model(self, model: str, strict: bool = False, output: Literal['table', 'name'] = 'table') -> pd.DataFrame:
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
            if output == 'name':
                return sorted(to_list(data["altLabel"].values))
            else:
                return data

    def _search_wmo_with(self, model: str, errors="raise"):
        """Return the list of WMOs with a given sensor model

        Notes
        -----
        Based on a fleet-monitoring API request to `platformCodes/multi-lines-search` on `sensorModels` field.
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
            if errors == 'raise':
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
        """Fetch and process JSON data returned from the fleet-monitoring API for a list of float WMOs

        Process float metadata (calibrations, sensors, cycles, configs, ...) for all WMOs with a given sensor model name

        Notes
        -----
        Based on fleet-monitoring API requests to `/floats/{wmo}`.
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

    def search(self, model: str, output: Literal['wmo', 'sn', 'wmo_sn'] = 'wmo', progress=False, errors='raise'):
        if output == 'wmo':
            return self._search_wmo_with(model=model, errors=errors)
        elif output == 'sn':
            return self._search_sn_with(model=model, progress=progress, errors=errors)
        elif output == 'wmo_sn':
            return self._search_wmo_sn_with(model=model, progress=progress, errors=errors)
        else:
            raise OptionValueError("'output' option value must be in: 'wmo', 'sn', or 'wmo_sn'")

    def iterfloats_with(self, model: str, chunksize: int = None):
        """Iterate over ArgoFloat equipped with a given sensor model

        By default, iterate over a single float, otherwise use the `chunksize` argument to iterate over chunk of floats.

        Parameters
        ----------
        model: str

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

            for float in ArgoSensor().iterfloats_with('SBE41CP'):
                float # is a ArgoFloat instance

        """
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

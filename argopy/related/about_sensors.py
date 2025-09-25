from typing import List

import pandas as pd

# from ..errors import InvalidOption
from ..stores import ArgoFloat, ArgoIndex, httpstore
from ..utils import check_wmo, Chunker, to_list
from ..errors import (
    DataNotFound,
    InvalidDataset,
    InvalidDatasetStructure,
    OptionValueError,
)
from ..options import OPTIONS
from . import ArgoNVSReferenceTables
import numpy as np


class ArgoSensor:
    """

    Notes
    -----
    We keep this class in line with:
        - https://github.com/OneArgo/ArgoVocabs/issues/156
        - https://github.com/OneArgo/ArgoVocabs/issues/157
    """

    def __init__(
        self,
        model: str = None,
        cache: bool = True,
        cachedir: str = "",
        timeout: int = 0,
    ):
        """Create an ArgoSensor helper class instance

        Parameters
        ----------
        model: str, optional
            A sensor model to use, by default None.

            Allowed values can be obtained with:
            ``ArgoSensor().reference_model_list['altLabel']``

        cache : bool, optional, default: False
            Use cache or not for fetched data
        cachedir: str, optional, default: OPTIONS['cachedir']
            Folder where to store cached files.
        timeout: int, optional, default: OPTIONS['api_timeout']
            Time out in seconds to connect to web API
        """
        self._cache = bool(cache)
        self._cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self.fs = httpstore(cache=self._cache, cachedir=self._cachedir)

        self._r25 = None
        self._r27 = None

        if model is not None:
            try:
                df = self.search_model(model, strict=True)
            except DataNotFound:
                raise DataNotFound(
                    f"No sensor model named '{model}', as per ArgoSensor().reference_model_list['altLabel'] values, based on Ref. Table 27."
                )

            if df.shape[0] == 1:
                self._model = model
                self._model_r27 = df.iloc[0].to_dict()
            else:
                raise InvalidDatasetStructure(
                    f"Found multiple sensor models with '{model}'. Refine your sensor model name to only one value in: {to_list(df['altLabel'].values)}"
                )

        else:
            self._model = None
            self._model_r27 = None

    @property
    def model(self):
        if self._model is not None:
            return self._model
        else:
            raise InvalidDataset(
                "No model name available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def model_long_name(self):
        if self._model_r27 is not None:
            return self._model_r27["prefLabel"]
        else:
            raise InvalidDataset(
                "No model long name available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def model_definition(self):
        if self._model_r27 is not None:
            return self._model_r27["definition"]
        else:
            raise InvalidDataset(
                "No model definition available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def model_deprecated(self):
        if self._model_r27 is not None:
            return self._model_r27["deprecated"]
        else:
            raise InvalidDataset(
                "No model deprecation available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def model_uri(self):
        if self._model_r27 is not None:
            return self._model_r27["id"]
        else:
            raise InvalidDataset(
                "No model URI available for an ArgoSensor instance not created with a specific sensor model"
            )

    def __repr__(self):
        if self._model_r27 is not None:
            summary = [f"<argosensor.{self.model}>"]
            summary.append(f"➤ {self.model_long_name}")
            if self.model_deprecated:
                summary.append("⛔ This model is deprecated !")
            else:
                summary.append("✅ This model is not deprecated.")
            summary.append(f"🔗 {self.model_uri}")
            summary.append(f"❝{self.model_definition}❞")
        else:
            summary = ["<argosensor>"]
            summary.append("This instance was not created with a sensor model name, you still have access to the following:")
            summary.append("👉 attributes: ")
            for attr in ['reference_model', 'reference_model_name', 'reference_sensor', 'reference_sensor_type']:
                summary.append(f"╰┈➤ ArgoSensor().{attr}")

            summary.append("👉 methods: ")
            for meth in ['search_model', 'search_model_name', 'search_wmo_with', 'search_sn_with', 'search_wmo_sn_with', 'iterfloats_with']:
                summary.append(f"╰┈➤ ArgoSensor().{meth}()")
        return "\n".join(summary)

    @property
    def reference_model(self) -> pd.DataFrame:
        """Return the official reference table for Argo sensor models

        Return the Argo Reference table R27 'SENSOR_MODEL':

        > Terms listing models of sensors mounted on Argo floats.
        """
        if self._r27 is None:
            self._r27 = ArgoNVSReferenceTables(cache=self._cache, cachedir=self._cachedir).tbl("R27")
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
            self._r25 = ArgoNVSReferenceTables(cache=self._cache, cachedir=self._cachedir).tbl("R25")
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

    def search_model(self, model: str, strict: bool = False) -> pd.DataFrame:
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
                raise DataNotFound(f"No sensor models matching '{model}'. You may try to search with strict=False.")
            else:
                raise DataNotFound(f"No sensor model names with '{model}' string occurrence.")
        else:
            return data

    def search_model_name(self, model: str = None, strict: bool = False) -> List[str]:
        """Return a list of Argo sensor model names matching a string

        Notes
        -----
        Argo netCDF variable SENSOR_MODEL is populated by such R27 altLabel names.
        """
        if model is None:
            if self._model is not None:
                model = self._model
            else:
                raise OptionValueError(
                    "You must provide a sensor model name or create an ArgoSensor instance with an exact sensor model name to use this method"
                )
        df = self.search_model(model=model, strict=strict)
        return sorted(to_list(df["altLabel"].values))

    def search_wmo_with(self, model: str):
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
            raise DataNotFound(f"No floats matching sensor model name '{model}'")
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
        Based on a fleet-monitoring API request to `/floats/{wmo}`.
        """
        wmos = self.search_wmo_with(model)

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

    def search_sn_with(self, model: str, progress=False, errors="raise"):
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

    def search_wmo_sn_with(self, model: str, progress=False, errors="raise"):
        """Return a dictionary of float WMOs with their sensor serial numbers"""

        def preprocess(jsdata, model_name: str = ""):
            sn = np.unique(
                [s["serial"] for s in jsdata["sensors"] if model_name in s["model"]]
            )
            return [jsdata["wmo"], sn]

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
        # from .. import ArgoFloat  # Prevent circular import

        wmos = self.search_wmo_with(model=model)

        idx = ArgoIndex(
            index_file="core",
            cache=self.cache,
        )

        if chunksize is not None:
            chk_opts = {}
            chk_opts.update({"chunks": {"wmo": "auto"}})
            chk_opts.update({"chunksize": {"wmo": chunksize}})
            chunked = Chunker(
                {"wmo": self.search_wmo_with(model=model)}, **chk_opts
            ).fit_transform()
            for grp in chunked:
                yield [ArgoFloat(wmo, idx=idx) for wmo in grp]

        else:
            for wmo in wmos:
                yield ArgoFloat(wmo, idx=idx)

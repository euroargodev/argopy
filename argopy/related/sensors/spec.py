import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal, Any, Iterator, Callable
import logging
import concurrent.futures
import xarray as xr
import logging
import warnings


from ...stores import ArgoFloat, ArgoIndex, httpstore, filestore
from ...stores.filesystems import (
    tqdm,
)  # Safe import, return a lambda if tqdm not available
from ...utils import check_wmo, Chunker, to_list, NVSrow, ppliststr, is_wmo
from ...errors import (
    DataNotFound,
    InvalidDataset,
    InvalidDatasetStructure,
    OptionValueError,
)
from ...options import OPTIONS
from ...utils import path2assets, register_accessor
from .. import ArgoNVSReferenceTables

from .references import SensorReferences, SensorModel, SensorType


log = logging.getLogger("argopy.related.sensors")

# Define allowed values as a tuple
SearchOutput = ("wmo", "sn", "wmo_sn", "df")
Error = ("raise", "ignore", "silent")

# Define Literal types using tuples
SearchOutputOptions = Literal[*SearchOutput]
ErrorOptions = Literal[*Error]


class ArgoSensor:

    __slots__ = [
        "_fs",
        "_cache",  # To cache extensions, not the option for filesystems
        "_model",
        "_type",
    ]

    def __init__(self, model: str | None = None, *args, **kwargs) -> None:
        if kwargs.get("fs", None) is None:
            self._fs = httpstore(
                cache=kwargs.get("cache", True),
                cachedir=kwargs.get("cachedir", OPTIONS["cachedir"]),
                timeout=kwargs.get("timeout", OPTIONS["api_timeout"]),
            )
        else:
            self._fs = kwargs["fs"]

        self._model: SensorModel | None = None
        self._type: SensorType | None = None
        if model is not None:
            try:
                df = self.ref.model.search(model)
            except DataNotFound:
                raise DataNotFound(
                    f"No sensor model named '{model}', as per ArgoSensor().ref.model.hint() values, based on Ref. Table 27."
                )

            if df.shape[0] == 1:
                self._model = SensorModel.from_series(df.iloc[0])
                self._type = self.ref.model.to_type(self._model, errors="ignore")
                # if "RBR" in self._model:
                # Add the RBR OEM API Authorization key for this sensor:
                # fs_kargs.update(client_kwargs={'headers': {'Authorization': OPTIONS.get('rbr_api_key') }})
            else:
                raise InvalidDatasetStructure(
                    f"Found multiple sensor models with '{model}'. Restrict your sensor model name to only one value in: {to_list(df['altLabel'].values)}"
                )

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
                "reference_manufacturer",
                "reference_manufacture_name",
            ]:
                summary.append(f"  ╰┈➤ ArgoSensor().{attr}")

            summary.append("👉 methods: ")
            for meth in [
                "search_model",
                "search",
                "iterfloats_with",
            ]:
                summary.append(f"  ╰┈➤ ArgoSensor().{meth}()")
        return "\n".join(summary)

    def _search_wmo_with(
        self, model: str | list[str], errors: ErrorOptions = "raise"
    ) -> list[int]:
        """Return the list of WMOs equipped with a given sensor model

        Notes
        -----
        Based on a fleet-monitoring API request to `platformCodes/multi-lines-search` on `sensorModels` field.

        Documentation:

        https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/platform-code-controller/getPlatformCodesMultiLinesSearchUsingPOST

        Notes
        -----
        No option checking, to be done by caller
        """
        models = to_list(model)
        api_endpoint = f"{OPTIONS['fleetmonitoring']}/platformCodes/multi-lines-search"
        payload = [
            {
                "nested": False,
                "path": "string",
                "searchValueType": "Text",
                "values": models,
                "field": "sensorModels",
            }
        ]
        wmos = self._fs.post(api_endpoint, json_data=payload)
        if wmos is None or len(wmos) == 0:
            if len(models) == 1:
                msg = f"Model is valid but no floats returned with this sensor model name: '{models[0]}'"
            else:
                msg = f"Models are valid but no floats returned with any of these sensor model names: {ppliststr(models)}"
            if errors == "raise":
                raise DataNotFound(msg)
            elif errors == "ignore":
                log.error(msg)
        return sorted(check_wmo(wmos))

    def _floats_api(
        self,
        model_or_wmo: str | int,
        preprocess: Callable = None,
        preprocess_opts: dict = {},
        postprocess: Callable = None,
        postprocess_opts: dict = {},
        progress: bool = False,
        errors: ErrorOptions = "raise",
    ) -> Any:
        """Search floats with a sensor model and then fetch and process JSON data returned from the fleet-monitoring API for each floats

        Notes
        -----
        Based on a POST request to the fleet-monitoring API requests to `/floats/{wmo}`.

        `Endpoint documentation <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/autonomous-float-controller/getFullFloatUsingGET>`_.

        Notes
        -----
        No option checking, to be done by caller
        """
        if preprocess_opts is None:
            preprocess_opts = {}

        try:
            is_wmo(model_or_wmo)
            WMOs = check_wmo(model_or_wmo)
        except ValueError:
            WMOs = self._search_wmo_with(model_or_wmo)

        URI = []
        for wmo in WMOs:
            URI.append(f"{OPTIONS['fleetmonitoring']}/floats/{wmo}")

        sns = self._fs.open_mfjson(
            URI,
            preprocess=preprocess,
            preprocess_opts=preprocess_opts,
            progress=progress,
            errors=errors,
            progress_unit="float",
            progress_desc="Fetching floats metadata",
        )

        if postprocess is not None:
            return postprocess(sns, **postprocess_opts)
        else:
            return sns

    def _search_sn_with(
        self,
        model: str,
        progress: bool = False,
        errors: ErrorOptions = "raise",
        **kwargs,
    ) -> list[str]:
        """Return serial number of sensor models with a given string in name

        Notes
        -----
        No option checking, to be done by caller
        """

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
            model if kwargs.get("wmo", None) is None else kwargs["wmo"],
            preprocess=preprocess,
            preprocess_opts={"model_name": model},
            postprocess=postprocess,
            progress=progress,
            errors=errors,
        )

    def _search_wmo_sn_with(
        self,
        model: str,
        progress: bool = False,
        errors: ErrorOptions = "raise",
        **kwargs,
    ) -> dict[int, str]:
        """Return a dictionary of float WMOs with their sensor serial numbers

        Notes
        -----
        No option checking, to be done by caller
        """

        def preprocess(jsdata, model_name: str = ""):
            sn = np.unique(
                [s["serial"] for s in jsdata["sensors"] if model_name in s["model"]]
            )
            return [jsdata["wmo"], [str(s) for s in sn]]

        def postprocess(data, **kwargs):
            S = {}
            for wmo, sn in data:
                S.update({check_wmo(wmo)[0]: to_list(sn)})
            return S

        results = self._floats_api(
            model if kwargs.get("wmo", None) is None else kwargs["wmo"],
            preprocess=preprocess,
            preprocess_opts={"model_name": model},
            postprocess=postprocess,
            progress=progress,
            errors=errors,
        )
        return dict(sorted(results.items()))

    def _to_dataframe(
        self,
        model: str,
        progress: bool = False,
        errors: ErrorOptions = "raise",
        **kwargs,
    ) -> pd.DataFrame:
        """Return a DataFrame with WMO, sensor type, model, maker, sn, units, accuracy and resolution

        Parameters
        ----------
        model: str, optional
            A string to search in the `sensorModels` field of the Euro-Argo fleet-monitoring API `platformCodes/multi-lines-search` endpoint.

        Notes
        -----
        No option checking, to be done by caller
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
                            "SerialNumber": sn if sn != "n/a" else None,
                            "Units": units,
                            "Accuracy": accuracy,
                            "Resolution": resolution,
                        }
                    )
            return pd.DataFrame(d).sort_values(by="WMO").reset_index(drop=True)

        df = self._floats_api(
            model if kwargs.get("wmo", None) is None else kwargs["wmo"],
            preprocess=preprocess,
            preprocess_opts={"model_name": model},
            postprocess=postprocess,
            progress=progress,
            errors=errors,
        )
        return df.sort_values(by="WMO", axis=0).reset_index(drop=True)

    def _search_single(
        self,
        model: str,
        output: SearchOutputOptions = "wmo",
        progress: bool = False,
        errors: ErrorOptions = "raise",
    ) -> list[int] | list[str] | dict[int, str] | pd.DataFrame:
        """Run a single model search"""

        if output == "df":
            return self._to_dataframe(model=model, progress=progress, errors=errors)
        elif output == "sn":
            return self._search_sn_with(model=model, progress=progress, errors=errors)
        elif output == "wmo_sn":
            return self._search_wmo_sn_with(
                model=model, progress=progress, errors=errors
            )
        else:
            return self._search_wmo_with(model=model, errors=errors)

    def _search_multi(
        self,
        models: list[str],
        output: SearchOutputOptions = "wmo",
        progress: bool = False,
        errors: ErrorOptions = "raise",
        max_workers: int | None = None,
    ) -> list[int] | list[str] | dict[int, str] | pd.DataFrame:
        """Run a multiple models search in parallel with multithreading"""
        # Remove duplicates:
        models = list(set(models))

        if output == "wmo":
            # Quite simple if we only need WMOs:
            return self._search_wmo_with(model=models, errors=errors)
        else:
            # Otherwise we need the list of WMOs to get metadata for
            # Even if we can request the fleetmonitoring API with multiple models at once, here we need
            # the tuple (model, wmo) to use multithreading and existing private search methods.
            lookup = []
            for model in models:
                wmos = self._search_wmo_with(model=model, errors=errors)
                lookup.extend([(model, wmo) for wmo in wmos])

        # For all other output format, we use multithreading to process all floats:
        if output == "sn":
            func = self._search_sn_with
        elif output == "wmo_sn":
            func = self._search_wmo_sn_with
        else:
            func = self._to_dataframe

        ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        failed = []
        if output in ["sn", "df"]:
            results = []
        elif output == "wmo_sn":
            results = {}

        with ConcurrentExecutor as executor:
            future_to_wmo = {
                executor.submit(
                    func,
                    pair[0],
                    errors=errors,
                    wmo=pair[1],
                ): pair
                for pair in lookup
            }
            futures = concurrent.futures.as_completed(future_to_wmo)
            if progress:
                futures = tqdm(
                    futures,
                    total=len(lookup),
                    disable="disable" in [progress],
                    unit="float",
                    desc=f"Fetching sensor meta-data for {len(lookup)} floats...",
                )

            for future in futures:
                data = None
                try:
                    data = future.result()
                except Exception:
                    failed.append(future_to_wmo[future])
                    if errors == "ignore":
                        log.error(
                            "Ignored error with this float: %s" % future_to_wmo[future]
                        )
                    elif errors == "silent":
                        pass
                    else:
                        raise
                finally:
                    # Gather results according to final output format:
                    if data is not None:
                        if output == "sn":
                            results.extend(data)
                        elif output == "df":
                            results.append(data)
                        elif output == "wmo_sn":
                            for wmo in data.keys():
                                if wmo in results:
                                    # Update existing key:
                                    results[wmo] = results[wmo].extend(data[wmo])
                                else:
                                    # Create new key:
                                    results.update({wmo: data[wmo]})

        if output != "wmo_sn":
            # Only keep non-empty results:
            results = [r for r in results if r is not None]
        else:
            results = dict(sorted(results.items()))

        if len(results) > 0:
            if output == "df":
                results = [r for r in results if r.shape[0] > 0]
                return (
                    pd.concat(results, axis=0)
                    .sort_values(by="WMO", axis=0)
                    .reset_index(drop=True)
                )
            else:
                return results
        raise DataNotFound(ppliststr(models))

    def search(
        self,
        model: str | list[str] | None = None,
        output: SearchOutputOptions = "wmo",
        progress: bool = True,
        errors: ErrorOptions = "raise",
        strict: bool = True,
        **kwargs,
    ) -> list[int] | list[str] | dict[int, str] | pd.DataFrame:
        """Search for Argo floats equipped with one or more sensor model name(s)

        All information are retrieved from the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu>`_.

        Since this method can return a large number of data, model names are expected to be exact. You can use the :meth:`ArgoSensor.ref.model.search` method to search for exact model names.

        Parameters
        ----------
        model: str, list[str], optional
            One or more exact model names to search data for.

        output: str, Literal["wmo", "sn", "wmo_sn", "df"], default "wmo"
            Define the output to return:

            - ``wmo``: a list of WMO numbers (integers)
            - ``sn``: a list of sensor serial numbers (strings)
            - ``wmo_sn``: a list of dictionary with WMO as key and serial numbers as values
            - ``df``: a :class:`pandas.DataFrame` with WMO, sensor type/model/maker and serial number

        progress: bool, default True
            Display a progress bar or not

        errors: str, Literal["raise", "ignore", "silent"], default "raise"
            Raise an error, log it or do nothing if the search return nothing.

        Returns
        -------
        list[int], list[str], dict[int, str], :class:`pandas.DataFrame`

        See Also
        --------
        :meth:`ArgoSensor.ref.model.search`

        Notes
        -----
        Whatever the output format, the first step is to retrieve a list of WMOs equipped with one or more sensor models.
        This is done using the Euro-Argo fleet-monitoring API and a request to the ``platformCodes/multi-lines-search`` endpoint using the ``sensorModels`` search field.

        Then if necessary (all output format but 'wmo'), the corresponding list of sensor serial numbers are retrieved using one request per float to the Euro-Argo fleet-monitoring API ``/floats/{wmo}`` endpoint.

        Web-api documentation:

        - `Documentation for endpoint: platformCodes/multi-lines-search <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/platform-code-controller/getPlatformCodesMultiLinesSearchUsingPOST>`_.
        - `Documentation for endpoint: /floats/{wmo} <https://fleetmonitoring.euro-argo.eu/swagger-ui.html#!/autonomous-float-controller/getFullFloatUsingGET>`_.

        """
        if output not in SearchOutput:
            raise OptionValueError(
                f"Invalid 'output' option value '{output}', must be in: {SearchOutput}"
            )
        if errors not in Error:
            raise OptionValueError(
                f"Invalid 'errors' option value '{errors}', must be in: {Error}"
            )

        if model is None:
            if self.model is not None:
                return self._search_single(
                    model=self.model.name,
                    output=output,
                    progress=progress,
                    errors=errors,
                )
            else:
                raise OptionValueError("You must specify at list one model to search !")

        models = to_list(model)

        def get_hints(these_models: str | list[str]):
            these_models = to_list(these_models)
            search_hint: list[str] = []
            for model in these_models:
                model = f"*{model.upper()}*"  # Use wildcards to get all possible hints
                try:
                    hint: list[str] = self.ref.model.search(
                        model,
                        output="name",
                    )
                    search_hint.extend(hint)
                except DataNotFound:
                    pass
            if len(search_hint) == 0:
                search_hint = ["No match !"]
            output = ppliststr(
                search_hint, last="or", n=20 if len(search_hint) > 20 else None
            )
            return output

        # Model names validation:
        valid_models: list[str] = []
        invalid_models: list[str] = []
        for model in models:
            if "*" in model:
                raise OptionValueError(
                    f"This method expect exact model names but got a '*' in '{model}'. Possible hints are: {get_hints(model)}"
                )
            try:
                hint = self.ref.model.search(model, output="name")
                valid_models.extend(hint)
            except DataNotFound:
                raise OptionValueError(
                    f"The model '{model}' is not exact. Possible hints are: {get_hints(model)}"
                )

        if len(valid_models) == 0:
            msg = f"Unknown sensor model name(s) {ppliststr(invalid_models)}. We expect exact sensor names such as: {get_hints(invalid_models)}."
            raise OptionValueError(msg)

        if len(valid_models) == 1:
            return self._search_single(
                model=valid_models[0], output=output, progress=progress, errors=errors
            )
        else:
            return self._search_multi(
                models=valid_models,
                output=output,
                progress=progress,
                errors=errors,
                **kwargs,
            )


@register_accessor("ref", ArgoSensor)
class References(SensorReferences):
    """An :class:`ArgoSensor` extension dedicated to reference tables appropriate for sensors

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoSensor

        ArgoSensor.ref.model.to_dataframe() # Return reference table R27 with the list of sensor models as a DataFrame
        ArgoSensor.ref.model.hint()      # Return list of sensor model names (possible values for 'SENSOR_MODEL' parameter)
        ArgoSensor.ref.model.to_type('SBE61') # Return sensor type (R25) of a given model (R27)

        ArgoSensor.ref.sensor.to_dataframe()  # Return reference table R25 with the list of sensor types as a DataFrame
        ArgoSensor.ref.sensor.hint()       # Return list of sensor types (possible values for 'SENSOR' parameter)
        ArgoSensor.ref.sensor.to_model('FLUOROMETER_CDOM') # Return all possible model names (R27) for a given sensor type (R25)

        ArgoSensor.ref.manufacturer.to_dataframe()  # Return reference table R26 with the list of manufacturer as a DataFrame
        ArgoSensor.ref.manufacturer.hint()       # Return list of manufacturer names (possible values for 'SENSOR_MAKER' parameter)

        ArgoSensor.ref.model.search('RBR') # Search for all (R27) referenced sensor models with some string in their name, return a DataFrame
        ArgoSensor.ref.model.search('RBR', output='name') # Return a list of names instead
        ArgoSensor.ref.model.search('SBE41CP', strict=False)
        ArgoSensor.ref.model.search('SBE41CP', strict=True)  # Exact string match required
    """

    _name = "ref"

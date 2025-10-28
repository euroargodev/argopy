import pandas as pd
import numpy as np
from typing import Literal, Any, Iterator, Callable
import concurrent.futures
import logging
import json
import sys

from ...stores import ArgoFloat, ArgoIndex, httpstore
from ...stores.filesystems import (
    tqdm,
)  # Safe import, return a lambda if tqdm not available
from ...utils import check_wmo, Chunker, to_list, ppliststr, is_wmo
from ...errors import (
    DataNotFound,
    InvalidDataset,
    InvalidDatasetStructure,
    OptionValueError,
)
from ...options import OPTIONS

from .references import SensorModel, SensorType


log = logging.getLogger("argopy.related.sensors")

# Define some options expected values as tuples
# (for argument validation)
SearchOutput = ("wmo", "sn", "wmo_sn", "df")
Error = ("raise", "ignore", "silent")
Ds = ("core", "deep", "bgc")

# Define Literal types using tuples
# (for typing)
SearchOutputOptions = Literal[*SearchOutput]
ErrorOptions = Literal[*Error]
DsOptions = Literal[*Ds]


class ArgoSensorSpec:

    __slots__ = [
        "_vocabulary",  # R27 row for an instance
        "_type",        # R25 row for an instance
        "_fs",  # http filesystem, extensions shall use it as well
        "_cache",  # To cache extensions, not the option for filesystems
    ]

    def __init__(self, model: str | None = None, *args, **kwargs) -> None:
        """Create an instance of :class:`ArgoSensor`

        Parameters
        ----------
        model: str, optional
            An exact sensor model name, None by default because this is optional.

            Otherwise, possible values can be obtained from :meth:`ArgoSensor.ref.model.hint`.

        Other Parameters
        ----------------
        fs: :class:`stores.httpstore`, default: None
            The http filesystem to use. If None is provided, we instantiate a new one based on `cache`, `cachedir` and `timeout` options.
        cache : bool, optional, default: True
            Use cache or not for fetched data. Used only if `fs` is None.
        cachedir: str, optional, default: OPTIONS['cachedir']
            Folder where to store cached files. Used only if `fs` is None.
        timeout: int, optional, default: OPTIONS['api_timeout']
            Time out in seconds to connect to web API. Used only if `fs` is None.

        """
        if kwargs.get("fs", None) is None:
            self._fs = httpstore(
                cache=kwargs.get("cache", True),
                cachedir=kwargs.get("cachedir", OPTIONS["cachedir"]),
                timeout=kwargs.get("timeout", OPTIONS["api_timeout"]),
            )
        else:
            self._fs = kwargs["fs"]

        self._vocabulary: SensorModel | None = None
        self._type: SensorType | None = None
        if model is not None:
            try:
                df = self.ref.model.search(model)
            except DataNotFound:
                raise DataNotFound(
                    f"No sensor model named '{model}', as per ArgoSensor().ref.model.hint() values, based on Ref. Table 27."
                )

            if df.shape[0] == 1:
                self._vocabulary = SensorModel.from_series(df.iloc[0])
                self._type = self.ref.model.to_type(self._vocabulary, errors="ignore")
                # if "RBR" in self._vocabulary:
                # Add the RBR OEM API Authorization key for this sensor:
                # fs_kargs.update(client_kwargs={'headers': {'Authorization': OPTIONS.get('rbr_api_key') }})
            else:
                raise InvalidDatasetStructure(
                    f"Found multiple sensor models with '{model}'. Restrict your sensor model name to only one value in: {to_list(df['altLabel'].values)}"
                )

    @property
    def vocabulary(self) -> SensorModel:
        """Argo reference "SENSOR_MODEL" vocabulary for this sensor model

        ! Only available for a class instance created with an explicit sensor model name.

        Returns
        -------
        :class:`SensorModel`

        Raises
        ------
        :class:`InvalidDataset`
        """
        if isinstance(self._vocabulary, SensorModel):
            return self._vocabulary
        else:
            raise InvalidDataset(
                "The 'vocabulary' property is not available for an ArgoSensor instance not created with a specific sensor model"
            )

    @property
    def type(self) -> SensorType:
        """Argo reference "SENSOR" vocabulary for this sensor model

        ! Only available for a class instance created with an explicit sensor model name.

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
        if isinstance(self._vocabulary, SensorModel):
            summary = [f"<argosensor.{self.type.name}.{self.vocabulary.name}>"]
            summary.append(f"TYPE➤ {self.type.long_name}")
            summary.append(f"MODEL➤ {self.vocabulary.long_name}")
            if self.vocabulary.deprecated:
                summary.append("⛔ This model is deprecated !")
            else:
                summary.append("✅ This model is not deprecated.")
            summary.append(f"🔗 {self.vocabulary.uri}")
            summary.append(f"❝{self.vocabulary.definition}❞")
        else:
            summary = ["<argosensor>"]
            summary.append(
                "This instance was not created with a sensor model name, you still have access to the following:"
            )
            summary.append("👉 extensions: ")
            for attr in [
                "ref.model.to_dataframe()",
                "ref.model.hint()",
                "ref.model.to_type",
                "ref.model.search",
                "ref.type.to_dataframe()",
                "ref.type.hint()",
                "ref.type.to_model",
                "ref.maker.to_dataframe()",
                "ref.maker.hint()",
            ]:
                summary.append(f"  ╰┈➤ ArgoSensor().{attr}")

            summary.append("👉 methods: ")
            for meth in [
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
            return np.sort(np.array(S)).tolist()

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
        if model is None and self.vocabulary is not None:
            model = self.vocabulary.name

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
        serialised: bool = False,
        **kwargs,
    ) -> list[int] | list[str] | dict[int, str] | pd.DataFrame | str:
        """Search for Argo floats equipped with one or more sensor model name(s)

        All information are retrieved from the `Euro-Argo fleet-monitoring API <https://fleetmonitoring.euro-argo.eu>`_.

        Since this method can return a large number of data, model names are expected to be exact. You can use the :meth:`ArgoSensor.ref.model.search` method to search for exact model names.

        Parameters
        ----------
        model: str, list[str], optional
            One or more exact model names to search data for.

        output: str, Literal["wmo", "sn", "wmo_sn", "df"], default: "wmo"
            Define the output to return:

            - ``wmo``: a list of WMO numbers (integers)
            - ``sn``: a list of sensor serial numbers (strings)
            - ``wmo_sn``: a list of dictionary with WMO as key and serial numbers as values
            - ``df``: a :class:`pandas.DataFrame` with WMO, sensor type/model/maker and serial number

        progress: bool, default: True
            Display a progress bar or not

        errors: str, Literal["raise", "ignore", "silent"], default: "raise"
            Raise an error, log it or do nothing for no search results.

        serialised: bool, default: False
            Return a serialised output. This allows for search results to be saved in cross-language, human-readable formats like json.

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
                f"Invalid 'output' option value '{output}', must be {ppliststr(SearchOutput, last='or')}"
            )
        if errors not in Error:
            raise OptionValueError(
                f"Invalid 'errors' option value '{errors}', must be in: {ppliststr(Error, last='or')}"
            )

        if model is None:
            if self.vocabulary is not None:
                return self._search_single(
                    model=self.vocabulary.name,
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
            results = self._search_single(
                model=valid_models[0], output=output, progress=progress, errors=errors
            )
        else:
            results = self._search_multi(
                models=valid_models,
                output=output,
                progress=progress,
                errors=errors,
                **kwargs,
            )
        if serialised:
            # Serialise all output format to json
            if output == 'df':
                return results.to_json(indent=2)
            return json.dumps(results, indent=2)
        return results

    def cli_search(self,
                   model: str | list[str] | None = None,
                   output: SearchOutputOptions = "wmo") -> str:  # type: ignore
        """A command-line-friendly search for Argo floats equipped with one or more sensor model name(s)

        This function is intended to call from the command-line and return serialized results for easy piping to other tools.

        Parameters
        ----------
        model: str, list[str], optional
            One or more exact model names to search data for.

        output: str, Literal["wmo", "sn", "wmo_sn", "df"], default: "wmo"
            Define the output to return:

            - ``wmo``: a list of WMO numbers (integers)
            - ``sn``: a list of sensor serial numbers (strings)
            - ``wmo_sn``: a list of dictionary with WMO as key and serial numbers as values
            - ``df``: a dictionary with 'WMO', 'Type', 'Model', 'Maker', 'SerialNumber', 'Units', 'Accuracy', 'Resolution' keys.

        Examples
        --------
        .. code-block:: bash
            :caption: Example of search results from the command-line

            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo')"
            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='sn')"
            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='wmo_sn')"
            python -c "from argopy import ArgoSensor; ArgoSensor().cli_search('RBR', output='df')"

        """
        print(self.search(model, output=output, serialised=True, progress=False), file=sys.stdout)

    def iterfloats_with(
        self,
        model: str | list[str] | None = None,
        chunksize: int | None = None,
        # ds: DsOptions = 'core',
        **kwargs
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

            sensors = ArgoSensor()

            for afloat in sensors.iterfloats_with("RAFOS"):
                print(afloat.WMO)

        """
        if model is None:
            if self.vocabulary is not None:
                model = self.vocabulary.name,
            else:
                raise OptionValueError("You must specify at list one model to search !")

        models = to_list(model)
        WMOs = self.search(model=models, progress=False)

        # 'ds' is a hidden option because I'm not 100% sure this will be needed.
        # ds: str, Literal['core', 'bgc', 'deep'], default='core'
        #     The Argo mission for this collection of floats.
        #     This will be used to create an :class:`ArgoIndex` shared by all :class:`ArgoFloat` instances.
        ds = kwargs.get('ds', 'core')
        if ds not in Ds:
            raise OptionValueError(
                f"Invalid 'ds' option value '{ds}', must be {ppliststr(Ds)}"
            )
        else:
            if ds == 'deep':
                ds = 'core'
            elif ds == 'bgc':
                ds = 'bgc-b'

        idx = ArgoIndex(
            index_file = ds,
            fs = self._fs,
        )

        if chunksize is not None:
            chk_opts = {}
            chk_opts.update({"chunks": {"wmo": "auto"}})
            chk_opts.update({"chunksize": {"wmo": chunksize}})
            chunked = Chunker(
                {"wmo": WMOs}, **chk_opts
            ).fit_transform()
            for grp in chunked:
                yield [ArgoFloat(wmo, idx=idx) for wmo in grp]

        else:
            for wmo in WMOs:
                yield ArgoFloat(wmo, idx=idx)

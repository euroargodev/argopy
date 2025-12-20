import pandas as pd
from pathlib import Path
from typing import Literal, NoReturn
from abc import ABC, abstractmethod
import logging
import fnmatch

from argopy.options import OPTIONS
from argopy.stores import httpstore, filestore
from argopy.related import ArgoNVSReferenceTables
from argopy.utils import to_list, path2assets, register_accessor, ppliststr
from argopy.errors import DataNotFound, OptionValueError

from argopy.related.sensors.accessories import SensorType, SensorModel
from argopy.related.sensors.accessories import Error, ErrorOptions


log = logging.getLogger("argopy.related.sensors.ref")


class SensorReferenceHolder(ABC):
    """Parent class to hold R25, R26, R27 and R27_to_R25 mapping data"""

    _r25: pd.DataFrame | None = None
    """NVS Reference table for Argo sensor types (R25)"""

    _r26: pd.DataFrame | None = None
    """NVS Reference table for Argo sensor maker (R26)"""

    _r27: pd.DataFrame | None = None
    """NVS Reference table for Argo sensor models (R27)"""

    _r27_to_r25: dict[str, str] | None = None
    """Dictionary mapping of R27 to R25"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError("A SensorReference instance cannot be called directly.")

    def __init__(self, obj):
        self._obj = obj  # An instance of SensorReferences, possibly with a filesystem
        if getattr(obj, "_fs", None) is None:
            self._fs = httpstore(
                cache=True,
                cachedir=OPTIONS["cachedir"],
                timeout=OPTIONS["api_timeout"],
            )
        else:
            self._fs = obj._fs

    @property
    def r25(self):
        """NVS Reference table for Argo sensor types (R25)"""
        if self._r25 is None:
            self._r25 = ArgoNVSReferenceTables(fs=self._fs).tbl("R25")
        return self._r25

    @property
    def r26(self):
        """NVS Reference table for Argo sensor maker (R26)"""
        if self._r26 is None:
            self._r26 = ArgoNVSReferenceTables(fs=self._fs).tbl("R26")
        return self._r26

    @property
    def r27(self):
        """NVS Reference table for Argo sensor models (R27)"""
        if self._r27 is None:
            self._r27 = ArgoNVSReferenceTables(fs=self._fs).tbl("R27")
        return self._r27

    def _load_mappers(self):
        """Load from static assets file the NVS R25 to R27 key mappings

        These mapping files were download from https://github.com/OneArgo/ArgoVocabs/issues/156.

        Column names: object_NVS_table, object_concept_id, predicate_code, subject_NVS_table, subject_concept_id, modification_type (I for Insertion)

        > A "predicate" indicates the relationship type between the "subject" and the "object"
        >
        > For "broader/narrower" relationship, the "BRD" predicate code is used
        >
        > For "related" relationship, the "MIN" predicate code is used (minor match)

        """
        df = []
        for p in (
            Path(path2assets).joinpath("vocabulary", "mapping", "R25_R27").glob("NVS_R25_R27_mappings_*.txt")
        ):
            df.append(
                filestore().read_csv(
                    p,
                    header=None,
                    names=["origin", "model", "predicate", "destination", "type", "modification"],
                )
            )
        df = pd.concat(df)
        for col in ['origin', 'destination', 'type', 'model', 'predicate', 'modification']:
            df[col] = df[col].apply(lambda x: x.strip())
        df = df.reset_index(drop=True)
        self._r27_to_r25 : pd.DataFrame | None = df

    @property
    def r27_to_r25(self) -> pd.DataFrame:
        """Dictionary mapping of R27 to R25"""
        if self._r27_to_r25 is None:
            self._load_mappers()
        return self._r27_to_r25

    @abstractmethod
    def to_dataframe(self):
        raise NotImplementedError

    @abstractmethod
    def hint(self):
        raise NotImplementedError


class SensorReferenceR27(SensorReferenceHolder):
    """Argo sensor models"""

    def to_dataframe(self) -> pd.DataFrame:
        """Reference Table **Sensor Models (R27)** as a :class:`pandas.DataFrame`

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgoNVSReferenceTables`
        """
        return self.r27

    def hint(self) -> list[str]:
        """List of Argo sensor models

        Return a sorted list of strings with altLabel from Argo Reference table R27 on 'SENSOR_MODEL'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR_MODEL`` is populated with values from this list.
        """
        return sorted(to_list(self.r27["altLabel"].values))

    def to_type(
        self,
        model: str | SensorModel,
        errors: ErrorOptions = "raise",
        obj: bool = False,
    ) -> list[str] | list[SensorType] | None:
        """Get all sensor types of a given sensor model

        All valid sensor model names can be obtained with :meth:`ArgoSensor.ref.model.hint`.

        Mapping between sensor model name (R27) and sensor type (R25) are from AVTT work at https://github.com/OneArgo/ArgoVocabs/issues/156.

        Parameters
        ----------
        model : str | :class:`argopy.related.SensorModel`
            The sensor model to read the sensor type for.
        errors : Literal["raise", "ignore", "silent"], optional, default: "raise"
            How to handle possible errors. If set to "ignore", the method may return None.
        obj: bool, optional, default: False
            Return a list of strings (False) or a list of :class:`argopy.related.SensorType`

        Returns
        -------
        list[str] | list[:class:`argopy.related.SensorType`] | None

        Raises
        ------
        :class:`DataNotFound`
        """
        if errors not in Error:
            raise OptionValueError(
                f"Invalid 'errors' option value '{errors}', must be in: {ppliststr(Error, last='or')}"
            )
        model_name: str = model.name if isinstance(model, SensorModel) else model

        match = fnmatch.filter(self.r27_to_r25["model"], model_name.upper())
        types = self.r27_to_r25[self.r27_to_r25["model"].apply(lambda x: x in match)]['type'].tolist()

        if len(types) > 0:
            # Since 'types' comes from the mapping, we double-check values against the R25 entries:
            types = [self.r25[self.r25["altLabel"].apply(lambda x: x == this_type)]['altLabel'].item() for this_type in types]
            if not obj:
                return types
            else:
                rows = [self.r25[self.r25["altLabel"].apply(lambda x: x == this_type)].iloc[0] for this_type in types]
                return [SensorType.from_series(row) for row in rows]

        elif errors == "raise":
            raise DataNotFound(
                f"Can't determine the type of sensor model '{model_name}' (no matching key in r27_to_r25 mapper)"
            )
        elif errors == "silent":
            log.error(
                f"Can't determine the type of sensor model '{model_name}' (no matching key in r27_to_r25 mapper)"
            )
        return None

    def search(
        self,
        model: str,
        output: Literal["df", "name"] = "df",
    ) -> pd.DataFrame | list[str]:
        """Return Argo sensor model references matching a string

        Look for occurrences in Argo Reference table R27 `altLabel` and return a subset of the :class:`pandas.DataFrame` with matching row(s).

        Parameters
        ----------
        model : str
            The model to search for. You can use wildcards: "SBE41CP*" "*DEEP*", "RBR*", or an exact name like "RBR_ARGO3_DEEP6".
        output : str, Literal["df", "name"], default "df"
            Is the output a :class:`pandas.DataFrame` with matching rows from R27, or a list of string.

        Returns
        -------
        :class:`pandas.DataFrame`, list[str]

        Raises
        ------
        :class:`DataNotFound`
        """
        match = fnmatch.filter(self.r27["altLabel"], model.upper())
        data = self.r27[self.r27["altLabel"].apply(lambda x: x in match)]

        if data.shape[0] == 0:
            raise DataNotFound(
                f"'{model}' is not a valid sensor model name. You can use wildcard for search, e.g. 'SBE61*'."
            )
        else:
            if output == "name":
                return sorted(to_list(data["altLabel"].values))
            else:
                return data.reset_index(drop=True)


class SensorReferenceR25(SensorReferenceHolder):
    """Argo sensor types"""

    def to_dataframe(self) -> pd.DataFrame:
        """Reference Table **Sensor Types (R25)** as a :class:`pandas.DataFrame`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        return self.r25

    def hint(self) -> list[str]:
        """List of Argo sensor types

        Return a sorted list of strings with altLabel from Argo Reference table R25 on 'SENSOR'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR`` is populated with values from this list.
        """
        return sorted(to_list(self.r25["altLabel"].values))

    def to_model(
        self,
        type: str | SensorType,
        errors: Literal["raise", "ignore"] = "raise",
        obj: bool = False,
    ) -> list[str] | list[SensorModel] | None:
        """Get all sensor model names of a given sensor type

        All valid sensor types can be obtained with :meth:`ArgoSensor.ref.sensor.hint`

        Mapping between sensor model name (R27) and sensor type (R25) are from AVTT work at https://github.com/OneArgo/ArgoVocabs/issues/156.

        Parameters
        ----------
        type : str, :class:`argopy.related.SensorType`
            The sensor type to read the sensor model name for.
        errors : Literal["raise", "ignore"] = "raise"
            How to handle possible errors. If set to "ignore", the method will return None.
        obj: bool, optional, default: False
            Return a list of strings (False) or a list of :class:`argopy.related.SensorModel`

        Returns
        -------
        list[str] | list[:class:`argopy.related.SensorModel`] | None

        Raises
        ------
        :class:`DataNotFound`
        """
        sensor_type = type.name if isinstance(type, SensorType) else type
        result = []

        match = fnmatch.filter(self.r27_to_r25["type"], sensor_type.upper())
        models = self.r27_to_r25[self.r27_to_r25["type"].apply(lambda x: x in match)]['model'].tolist()

        if len(models) > 0:
            # Since 'models' comes from the mapping, we double-check values against the R27 entries:
            models = [self.r27[self.r27["altLabel"].apply(lambda x: x == model)]['altLabel'].item() for
             model in models]
            if not obj:
                return models
            else:
                rows = [self.r27[self.r27["altLabel"].apply(lambda x: x == model)].iloc[0] for
             model in models]
                return [SensorModel.from_series(row) for row in rows]

        if len(result) == 0:
            if errors == "raise":
                raise DataNotFound(
                    f"Can't find any sensor model for this type '{sensor_type}' (no matching key in r27_to_r25 mapper)"
                )
            else:
                return None
        else:
            return result


class SensorReferenceR26(SensorReferenceHolder):
    """Argo sensor maker"""

    def to_dataframe(self) -> pd.DataFrame:
        """Reference Table **Sensor Makers (R26)** as a :class:`pandas.DataFrame`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        return self.r26

    def hint(self) -> list[str]:
        """List of Argo sensor makers

        Return a sorted list of strings with altLabel from Argo Reference table R26 on 'SENSOR_MAKER'.

        Returns
        -------
        list[str]

        Notes
        -----
        Argo netCDF variable ``SENSOR_MAKER`` is populated with values from this list.
        """
        return sorted(to_list(self.r26["altLabel"].values))


class SensorReferences:

    __slots__ = ["_obj", "_fs"]

    def __init__(self, obj):
        self._obj = obj  # An instance of ArgoSensor, possibly with a filesystem
        if getattr(obj, "_fs", None) is None:
            self._fs = httpstore(
                cache=True,
                cachedir=OPTIONS["cachedir"],
                timeout=OPTIONS["api_timeout"],
            )
        else:
            self._fs = obj._fs

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError("ArgoSensor.ref cannot be called directly.")


@register_accessor("type", SensorReferences)
class SensorExtension(SensorReferenceR25):
    _name = "ref.type"


@register_accessor("maker", SensorReferences)
class MakerExtension(SensorReferenceR26):
    _name = "ref.maker"


@register_accessor("model", SensorReferences)
class ModelExtension(SensorReferenceR27):
    _name = "ref.model"

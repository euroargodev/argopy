import pandas as pd
from pathlib import Path
from typing import Literal, NoReturn
from abc import ABC, abstractmethod
import logging
import fnmatch

from ...options import OPTIONS
from ...stores import httpstore, filestore
from ...related import ArgoNVSReferenceTables
from ...utils import to_list, NVSrow, path2assets, register_accessor
from ...errors import DataNotFound


log = logging.getLogger("argopy.related.sensors.ref")


# Define allowed values as a tuple
Error = ("raise", "ignore", "silent")

# Define Literal types using tuples
ErrorOptions = Literal[*Error]


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
        return (
            string.lower() in self.name.lower()
            or string.lower() in self.long_name.lower()
        )


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
        raise ValueError(
            "A SensorReference instance cannot be called directly."
        )

    def __init__(self, obj):
        self._obj = obj  # An instance of SensorReferences, possibly with a filesystem
        if getattr(obj, '_fs', None) is None:
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
    def r27_to_r25(self):
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
        """Official reference table for Argo sensor models (R27)

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgoNVSReferenceTables`
        """
        return self.r27

    def hint(self) -> list[str]:
        """Official list of Argo sensor models (R27)

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
        model: str | SensorModel | None = None,
        errors: ErrorOptions = "raise",
    ) -> SensorType | None:
        """Get a sensor type for a given sensor model

        All valid sensor model name can be obtained with :meth:`ArgoSensor.ref.mode.to_list()`.

        Mapping between sensor model name (R27) and sensor type (R25) are from AVTT work at https://github.com/OneArgo/ArgoVocabs/issues/156.

        Parameters
        ----------
        model : str | :class:`SensorModel`
            The model to read the sensor type for.
        errors : Literal["raise", "ignore", "silent"] = "raise"
            How to handle possible errors. If set to "ignore", the method will return None.

        Returns
        -------
        :class:`SensorType` | None
        """
        model_name: str = model.name if isinstance(model, SensorModel) else model
        sensor_type = self.r27_to_r25.get(model_name, None)
        if sensor_type is not None:
            row = self.r25[
                self.r25["altLabel"].apply(lambda x: x == sensor_type)
            ].iloc[0]
            return SensorType.from_series(row)
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

        Look for occurrences in Argo Reference table R27 `altLabel` and return a :class:`pandas.DataFrame` with matching row(s).

        Parameters
        ----------
        model : str
            The model to search for. You can use wildcards: "SBE41CP*" "*DEEP*", "RBR*", or an exact name like "RBR_ARGO3_DEEP6".
        output : str, Literal["df", "name"], default "df"
            Is the output a :class:`pandas.DataFrame` with matching rows from :attr:`ArgoSensor.reference_model`, or a list of string.

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
        """Official reference table for Argo sensor types (R25)

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        return self.r25

    def hint(self) -> list[str]:
        """Official list of Argo sensor types (R25)

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
        """
        sensor_type = type.name if isinstance(type, SensorType) else type
        result = []
        for key, val in self.r27_to_r25.items():
            if sensor_type.lower() in val.lower():
                row = self.r27[
                    self.r27["altLabel"].apply(lambda x: x == key)
                ].iloc[0]
                result.append(SensorModel.from_series(row).name)
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
        """Official reference table for Argo sensor makers (R26)

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        return self.r26

    def hint(self) -> list[str]:
        """Official list of Argo sensor maker (R26)

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
        if getattr(obj, '_fs', None) is None:
            self._fs = httpstore(
                cache=True,
                cachedir=OPTIONS["cachedir"],
                timeout=OPTIONS["api_timeout"],
            )
        else:
            self._fs = obj._fs

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoSensor.ref cannot be called directly."
        )


@register_accessor('type', SensorReferences)
class SensorExtension(SensorReferenceR25):
    _name = "ref.type"

@register_accessor('maker', SensorReferences)
class MakerExtension(SensorReferenceR26):
    _name = "ref.maker"

@register_accessor('model', SensorReferences)
class ModelExtension(SensorReferenceR27):
    _name = "ref.model"

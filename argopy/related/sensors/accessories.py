from typing import Literal, Any
import pandas as pd
import xarray as xr
import numpy as np
from dataclasses import dataclass
from html import escape

from argopy.options import OPTIONS
from argopy.utils import ppliststr, to_list
from argopy.utils import NVSrow
from argopy.utils.schemas.sensors.spec import Parameter, Sensor, SensorInfo
from argopy.related.sensors.oem.oem_metadata_repr import OemMetaDataDisplay


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


class SensorType(NVSrow):
    """One single sensor type data from a R25-"Argo sensor types" row

    .. warning::
        This class is experimental and may change in a future release.

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

    .. warning::
        This class is experimental and may change in a future release.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoSensor

        sm = ArgoSensor('AANDERAA_OPTODE_4330F').vocabulary

        sm.name
        sm.long_name
        sm.definition
        sm.deprecated
        sm.urn
        sm.uri

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
        sm.urn
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


@dataclass
class SensorModelMetaData:
    parameters: list[Parameter]
    sensors: list[Sensor]
    sensor_info: SensorInfo
    instrument_vendorinfo: Any


class SensorMetaData:
    """A placeholder for float sensors meta-data

    This is a design in active dev. No final specs.

    ..code-block:: python
        meta = ArgoFloat(WMO).open_dataset('meta')
        md = SensorMetaData(meta)

        md.param2sensor  # Dictionary mapping PARAMETER to PARAMETER_SENSOR
        md['DOXY']  # Dictionary with all PARAMETER meta-data
    """

    __slots__ = (
        "_obj",
        "_param2sensor",
        "_wmo",
        "_data",
        "_parameters",
        "_sensors",
        "_models",
        "sensor_info",
        "instrument_vendorinfo",
        "sensors",
        "parameters",
    )

    def __init__(self, obj):
        self._obj: xr.Dataset = obj
        self._param2sensor: dict[str, str] = None
        self._wmo: int = self._obj["PLATFORM_NUMBER"].item()
        self._data: dict[str, Any] = {}

        self._parameters: list[str] = sorted(list(set(self.param2sensor.keys())))
        self._sensors: list[str] = sorted(list(set(self.param2sensor.values())))
        self._models: list[str] = sorted(
            set([self.sensor2dict(s)["SENSOR_MODEL"] for s in self._sensors])
        )

        self.parameters: list[Parameter] = [self.to_schema(p) for p in self._parameters]
        self.sensors: list[Sensor] = [self.to_schema(p) for p in self._sensors]
        self.sensor_info: SensorInfo = SensorInfo(
            **{
                "created_by": self._obj.attrs["institution"],
                "date_creation": self._obj.attrs["history"]
                .split(";")[0]
                .strip()
                .split(" ")[0],
                "link": "./argo.sensor.schema.json",
                "format_version": self._obj.attrs["Conventions"],
                "contents": "",
                "sensor_described": "",
            }
        )
        self.instrument_vendorinfo = None

    def __repr__(self):
        summary = [f"<argofloat.{self._wmo}.sensormetadata.file>"]
        n_model, n_param, n_sensor = (
            len(self._models),
            len(self._parameters),
            len(self._sensors),
        )
        summary.append(
            f"> {n_model} sensor models, equiped with {n_sensor} sensor types and providing {n_param} parameters"
        )
        summary.append(f"models: {ppliststr(self._models)}")
        summary.append(f"sensors: {ppliststr(self._sensors)}")
        summary.append(f"parameters: {ppliststr(self._parameters)}")
        return "\n".join(summary)

    @property
    def param2sensor(self):
        """Dictionary mapping PARAMETER to PARAMETER_SENSOR"""
        if self._param2sensor is None:

            def get(parameter):
                parameter = parameter.strip()
                plist = list(self._obj["PARAMETER"].values)
                iparams = {}
                [iparams.update({param.strip(): plist.index(param)}) for param in plist]
                if parameter in iparams.keys():
                    # PARAMETER_SENSOR (N_PARAM): name, populated from R25, of the sensor measuring this parameter
                    val = (
                        self._obj["PARAMETER_SENSOR"]
                        .isel({"N_PARAM": iparams[parameter]})
                        .values[np.newaxis][0]
                        .strip()
                    )
                    return val
                return None

            result = {}
            for param in self._obj["PARAMETER"]:
                p = param.values[np.newaxis][0]
                result.update({str(p).strip(): get(p)})
            self._param2sensor = result
        return self._param2sensor

    def coef2dict(self, text: str) -> list[dict[str, Any]]:
        """Transform a calibration coefficient string into a list of dictionaries"""
        try:
            result = []
            for coef_grp in text.split(";"):
                grp = {}
                for coef in coef_grp.split(","):
                    grp.update(
                        {
                            k.strip(): float(v)
                            for k, v in (
                                pair.split("=") for pair in coef.strip().split(",")
                            )
                        }
                    )
                result.append(grp)
            return result
        except:
            return text.strip()

    def sensor2dict(self, sensor: str) -> dict[str, Any]:
        """Extract one sensor data from a metadata xr.dataset"""
        sensor = sensor.strip()
        slist = list(self._obj["SENSOR"].values)
        isensors = {}
        [isensors.update({s.strip(): slist.index(s)}) for s in slist]
        result = {}
        if sensor in isensors.keys():
            for var in self._obj.data_vars:
                if self._obj[var].dims == ("N_SENSOR",):
                    val = (
                        self._obj[var]
                        .isel({"N_SENSOR": isensors[sensor]})
                        .values[np.newaxis][0]
                        .strip()
                    )
                    val = None if val in ["none", b""] else val
                    result.update({var: val})
            return result
        return None

    def param2dict(self, parameter: str) -> dict[str, Any]:
        """Extract one parameter data from a metadata xr.dataset"""
        parameter = parameter.strip()
        plist = list(self._obj["PARAMETER"].values)
        iparams = {}
        [iparams.update({param.strip(): plist.index(param)}) for param in plist]
        result = {}
        if parameter in iparams.keys():
            for var in self._obj.data_vars:
                if self._obj[var].dims == ("N_PARAM",):
                    val = (
                        self._obj[var]
                        .isel({"N_PARAM": iparams[parameter]})
                        .values[np.newaxis][0]
                        .strip()
                    )
                    if var == "PARAMETER_SENSOR":
                        val = self.sensor2dict(val)
                    if var == "PREDEPLOYMENT_CALIB_COEFFICIENT":
                        val = self.coef2dict(val)
                    val = None if val in ["none", b""] else val
                    result.update({var: val})
            return result
        return None

    def to_schema(self, key: str) -> Parameter | Sensor:
        """Return parameter or sensor data as a JSON-schema compliant object"""

        def param2sdn(param):
            pd = self[param]
            spd = pd.copy()
            spd["PARAMETER"] = f"SDN:R03::{pd['PARAMETER']}"
            spd["PARAMETER_SENSOR"] = f"SDN:R25::{pd['PARAMETER_SENSOR']['SENSOR']}"
            spd["PREDEPLOYMENT_CALIB_COEFFICIENT_LIST"] = spd[
                "PREDEPLOYMENT_CALIB_COEFFICIENT"
            ]
            spd.pop("PREDEPLOYMENT_CALIB_COEFFICIENT")
            spd["PREDEPLOYMENT_CALIB_DATE"] = "none"
            return Parameter(**spd)

        def sensor2sdn(sensor):
            pd = self[sensor]
            spd = pd.copy()
            spd["SENSOR"] = f"SDN:R25::{pd['SENSOR']}"
            spd["SENSOR_MAKER"] = f"SDN:R26::{pd['SENSOR_MAKER']}"
            spd["SENSOR_MODEL"] = f"SDN:R27::{pd['SENSOR_MODEL']}"
            return Sensor(**spd)

        if key in self.param2sensor.keys():
            return param2sdn(key)
        else:
            return sensor2sdn(key)

    def __getitem__(self, key: str) -> dict[str, Any]:
        """Get parameter or sensor data as dictionary"""
        if self._data.get(key, None) is None:
            if key in self.param2sensor.keys():
                self._data.update({key: self.param2dict(key)})
            elif key in self.param2sensor.values():
                self._data.update({key: self.sensor2dict(key)})
            else:
                raise ValueError(f"Unknown parameter or sensor '{key}'")
        return self._data[key]

    @property
    def Models(self):
        # Models = {'<model>': {'sensors': [], 'parameters': []}}
        Models = {}
        for model in self._models:
            # List of sensor types:
            Models.update({model: {"sensors": [], "parameters": []}})

        for sensor in self._sensors:
            model = self[sensor]["SENSOR_MODEL"]
            if self.to_schema(sensor) not in Models[model]["sensors"]:
                Models[model]["sensors"].append(self.to_schema(sensor))

        for parameter in self._parameters:
            model = self[parameter]["PARAMETER_SENSOR"]["SENSOR_MODEL"]
            if self.to_schema(parameter) not in Models[model]["parameters"]:
                Models[model]["parameters"].append(self.to_schema(parameter))

        return Models

    def display_model(self, model: str | None = None):
        if model not in self.Models.keys():
            raise ValueError(f"Model name must be one in {ppliststr(self.Models.keys())}")

        this_model = SensorModelMetaData(
            **{
                "sensors": self.Models[model]["sensors"],
                "parameters": self.Models[model]["parameters"],
                "sensor_info": SensorInfo(
                    **{
                        "created_by": self._obj.attrs["institution"],
                        "date_creation": self._obj.attrs["history"]
                        .split(";")[0]
                        .strip()
                        .split(" ")[0],
                        "link": "./argo.sensor.schema.json",
                        "format_version": self._obj.attrs["Conventions"],
                        "contents": "",
                        "sensor_described": model,
                    }
                ),
                "instrument_vendorinfo": "",
            }
        )
        oem_like = OemMetaDataDisplay(this_model)

        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(oem_like))}</pre>"

        from IPython.display import display, HTML
        display(HTML(oem_like.html))

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from jsonschema import validate, ValidationError

from ...stores import httpstore
from ...options import OPTIONS
from ...utils import urnparser
from .oem_metadata_repr import NotebookCellDisplay


@dataclass
class SensorInfo:
    created_by: str
    date_creation: str  # ISO 8601 datetime string
    link: str
    format_version: str
    contents: str
    sensor_described: str


@dataclass
class Context:
    SDN_R03: str = "http://vocab.nerc.ac.uk/collection/R03/current/"
    SDN_R25: str = "http://vocab.nerc.ac.uk/collection/R25/current/"
    SDN_R26: str = "http://vocab.nerc.ac.uk/collection/R26/current/"
    SDN_R27: str = "http://vocab.nerc.ac.uk/collection/R27/current/"
    SDN_L22: str = "http://vocab.nerc.ac.uk/collection/L22/current/"


@dataclass
class Sensor:
    SENSOR: str  # SDN:R25::CTD_PRES
    SENSOR_MAKER: str  # SDN:R26::RBR
    SENSOR_MODEL: str  # SDN:R27::RBR_PRES_A
    SENSOR_FIRMWARE_VERSION: str  # wrong key used by RBR, https://github.com/euroargodev/sensor_metadata_json/issues/20
    # SENSOR_MODEL_FIRMWARE: str # Correct schema key
    SENSOR_SERIAL_NO: str
    sensor_vendorinfo: Optional[Dict[str, Any]] = None

    @property
    def SENSOR_uri(self):
        urnparts = urnparser(self.SENSOR)
        return f"{OPTIONS['nvs']}/{urnparts['listid']}/current/{urnparts['termid']}"

    @property
    def SENSOR_MAKER_uri(self):
        urnparts = urnparser(self.SENSOR_MAKER)
        return f"{OPTIONS['nvs']}/{urnparts['listid']}/current/{urnparts['termid']}"

    @property
    def SENSOR_MODEL_uri(self):
        urnparts = urnparser(self.SENSOR_MODEL)
        return f"{OPTIONS['nvs']}/{urnparts['listid']}/current/{urnparts['termid']}"


@dataclass
class Parameter:
    PARAMETER: str  # SDN:R03::PRES
    PARAMETER_SENSOR: str  # SDN:R25::CTD_PRES
    PARAMETER_UNITS: str
    PARAMETER_ACCURACY: str
    PARAMETER_RESOLUTION: str
    PREDEPLOYMENT_CALIB_EQUATION: str
    PREDEPLOYMENT_CALIB_COEFFICIENT_LIST: Dict[str, str]
    PREDEPLOYMENT_CALIB_COMMENT: str
    PREDEPLOYMENT_CALIB_DATE: str
    parameter_vendorinfo: Optional[Dict[str, Any]] = None
    predeployment_vendorinfo: Optional[Dict[str, Any]] = None

    @property
    def PARAMETER_uri(self):
        urnparts = urnparser(self.PARAMETER)
        return f"{OPTIONS['nvs']}/{urnparts['listid']}/current/{urnparts['termid']}"

    @property
    def PARAMETER_SENSOR_uri(self):
        urnparts = urnparser(self.PARAMETER_SENSOR)
        return f"{OPTIONS['nvs']}/{urnparts['listid']}/current/{urnparts['termid']}"


def fetch_rbr_data(sn: str, **kwargs):
    if kwargs.get("fs", None) is None:
        _cache = kwargs.get("cache", True)
        _cachedir = kwargs.get("cachedir", OPTIONS["cachedir"])
        _timeout = kwargs.get("timeout", OPTIONS["api_timeout"])
        fs_kargs = {"cache": _cache, "cachedir": _cachedir, "timeout": _timeout}
        _api_key = kwargs.get("rbr_api_key", OPTIONS["rbr_api_key"])
        fs_kargs.update(client_kwargs={"headers": {"Authorization": _api_key}})
        fs = httpstore(**fs_kargs)
    uri = f"{OPTIONS['rbr_api']}/instruments/{sn}/argometadatajson"
    return fs.open_json(uri)


class OemArgoSensorMetaData:
    """Argo sensor meta-data from OEM

    OEM : Original Equipment Manufacturer

    Comply to schema from https://github.com/euroargodev/sensor_metadata_json

    Examples
    --------
    .. code-block:: python

        OemArgoSensorMetaData()

        OemArgoSensorMetaData(validate=True)  # Run json schema validation compliance automatically

        OemArgoSensorMetaData().from_dict(jsdata)  # Use any compliant json data from OEM

        OemArgoSensorMetaData.from_rbr(208380)  # Direct call to the RBR api


    """

    _schema_src = "https://raw.githubusercontent.com/euroargodev/sensor_metadata_json/refs/heads/main/schemas/argo.sensor.schema.json"
    """URI of the argo sensor JSON schema"""

    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        validate: bool = False,
        **kwargs,
    ):
        self._run_validation = validate
        if self._run_validation:
            self.schema = self._read_schema()

        self.sensor_info: Optional[SensorInfo] = None
        self.context: Optional[Context] = None
        self.sensors: List[Sensor] = field(default_factory=list)
        self.parameters: List[Parameter] = field(default_factory=list)
        self.instrument_vendorinfo: Optional[Dict[str, Any]] = None

        if kwargs.get("fs", None) is not None:
            self._fs = kwargs.get("fs")
        else:
            self._cache = kwargs.get("cache", True)
            self._cachedir = kwargs.get("cachedir", OPTIONS["cachedir"])
            self._timeout = kwargs.get("timeout", OPTIONS["api_timeout"])
            fs_kargs = {
                "cache": self._cache,
                "cachedir": self._cachedir,
                "timeout": self._timeout,
            }
            self._fs = httpstore(**fs_kargs)

        if json_data:
            self.from_dict(json_data)

    def __repr__(self):
        sensor_count = len(self.sensors)
        parameter_count = len(self.parameters)
        sensor_described = (
            self.sensor_info.sensor_described if self.sensor_info else "N/A"
        )
        created_by = self.sensor_info.created_by if self.sensor_info else "N/A"
        date_creation = self.sensor_info.date_creation if self.sensor_info else "N/A"

        summary = [f"<oemsensor.{sensor_described}>"]
        summary.append(f"created_by: '{created_by}'")
        summary.append(f"date_creation: '{date_creation}'")
        summary.append(f"sensors: {sensor_count} {[urnparser(s.SENSOR)['termid'] for s in self.sensors]}")
        summary.append(f"parameters: {parameter_count} {[urnparser(s.PARAMETER)['termid'] for s in self.parameters]}")
        summary.append(
            f"instrument_vendorinfo: {'Present' if self.instrument_vendorinfo else 'None'}"
        )
        return "\n".join(summary)

    def _repr_html_(self):
        return NotebookCellDisplay(self).html

    def _ipython_display_(self):
        from IPython.display import display, HTML

        display(HTML(NotebookCellDisplay(self).html))

    def _read_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        schema = httpstore().open_json(self.schema_src)
        return schema

    def from_dict(self, data: Dict[str, Any]):
        """Load data from a dictionary and validate it."""
        if self._run_validation:
            try:
                validate(instance=data, schema=self.schema)
            except ValidationError as e:
                raise ValueError(f"Json schema Validation error: {e.message}")

        self.sensor_info = SensorInfo(**data["sensor_info"])
        self.context = Context(
            **{
                k.replace("::", "").replace(":", "_"): v
                for k, v in data["@context"].items()
            }
        )
        self.sensors = [Sensor(**sensor) for sensor in data["SENSORS"]]
        self.parameters = [Parameter(**param) for param in data["PARAMETERS"]]
        self.instrument_vendorinfo = data.get("instrument_vendorinfo")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object back to a dictionary, following schema"""
        return {
            "sensor_info": {
                "created_by": self.sensor_info.created_by,
                "date_creation": self.sensor_info.date_creation,
                "link": self.sensor_info.link,
                "format_version": self.sensor_info.format_version,
                "contents": self.sensor_info.contents,
                "sensor_described": self.sensor_info.sensor_described,
            },
            "@context": {
                "SDN:R03::": self.context.SDN_R03,
                "SDN:R25::": self.context.SDN_R25,
                "SDN:R26::": self.context.SDN_R26,
                "SDN:R27::": self.context.SDN_R27,
                "SDN:L22::": self.context.SDN_L22,
            },
            "SENSORS": [
                {
                    "SENSOR": sensor.SENSOR,
                    "SENSOR_MAKER": sensor.SENSOR_MAKER,
                    "SENSOR_MODEL": sensor.SENSOR_MODEL,
                    "SENSOR_MODEL_FIRMWARE": getattr(
                        sensor, "SENSOR_MODEL_FIRMWARE", sensor.SENSOR_FIRMWARE_VERSION
                    ),
                    "SENSOR_SERIAL_NO": sensor.SENSOR_SERIAL_NO,
                    "sensor_vendorinfo": sensor.sensor_vendorinfo,
                }
                for sensor in self.sensors
            ],
            "PARAMETERS": [
                {
                    "PARAMETER": param.PARAMETER,
                    "PARAMETER_SENSOR": param.PARAMETER_SENSOR,
                    "PARAMETER_UNITS": param.PARAMETER_UNITS,
                    "PARAMETER_ACCURACY": param.PARAMETER_ACCURACY,
                    "PARAMETER_RESOLUTION": param.PARAMETER_RESOLUTION,
                    "PREDEPLOYMENT_CALIB_EQUATION": param.PREDEPLOYMENT_CALIB_EQUATION,
                    "PREDEPLOYMENT_CALIB_COEFFICIENT_LIST": param.PREDEPLOYMENT_CALIB_COEFFICIENT_LIST,
                    "PREDEPLOYMENT_CALIB_COMMENT": param.PREDEPLOYMENT_CALIB_COMMENT,
                    "PREDEPLOYMENT_CALIB_DATE": param.PREDEPLOYMENT_CALIB_DATE,
                    "parameter_vendorinfo": param.parameter_vendorinfo,
                    "predeployment_vendorinfo": param.predeployment_vendorinfo,
                }
                for param in self.parameters
            ],
            "instrument_vendorinfo": self.instrument_vendorinfo,
        }

    def save_to_json_file(self, file_path: str) -> None:
        """Save the object to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_rbr(cls, serial_number: str, **kwargs):
        """Fetch sensor metadata from RBR API and return an OemArgoSensorMetaData instance"""
        # Use your HTTP store or API client to fetch data
        data = fetch_rbr_data(serial_number, **kwargs)
        return cls(json_data=data, **kwargs)

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from jsonschema import validate, ValidationError

from ...stores import httpstore
from ...options import OPTIONS
from ...utils import urnparser
from .oem_metadata_repr import OemMetaDataDisplay, ParameterDisplay


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

    def __repr__(self):
        summary = [f"<oemsensor.sensor.{self.SENSOR_SERIAL_NO}>"]
        summary.append(f"  SENSOR: {self.SENSOR} ({self.SENSOR_uri})")
        summary.append(f"  SENSOR_MAKER: {self.SENSOR_MAKER} ({self.SENSOR_MAKER_uri})")
        summary.append(f"  SENSOR_MODEL: {self.SENSOR_MODEL} ({self.SENSOR_MODEL_uri})")
        summary.append(f"  SENSOR_FIRMWARE_VERSION: {self.SENSOR_FIRMWARE_VERSION}")
        summary.append(f"  sensor_vendorinfo:")
        for key in self.sensor_vendorinfo.keys():
            summary.append(f"    - {key}: {self.sensor_vendorinfo[key]}")
        return "\n".join(summary)


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

    def __repr__(self):
        summary = [f"<oemsensor.parameter.{self.PARAMETER}>"]
        summary.append(f"  PARAMETER: {self.PARAMETER} ({self.PARAMETER_uri})")
        summary.append(f"  PARAMETER_SENSOR: {self.PARAMETER_SENSOR} ({self.PARAMETER_SENSOR_uri})")
        for key in ['UNITS', 'ACCURACY', 'RESOLUTION']:
            p = f"PARAMETER_{key}"
            summary.append(f"  {key}: {getattr(self, p, 'N/A')}")
        for key in ['EQUATION', 'COEFFICIENT', 'COMMENT', 'DATE']:
            p = f"PREDEPLOYMENT_CALIB_{key}"
            summary.append(f"  {key}: {getattr(self, p, 'N/A')}")
        for key in ['parameter_vendorinfo', 'predeployment_vendorinfo']:
            summary.append(f"  {key}: {getattr(self, key, 'N/A')}")
        return "\n".join(summary)

    def _repr_html_(self):
        return ParameterDisplay(self).html

    def _ipython_display_(self):
        from IPython.display import display, HTML
        display(HTML(ParameterDisplay(self).html))

class ArgoSensorMetaDataOem:
    """Argo sensor meta-data - from OEM

    A class helper to work with meta-data structure complying to schema from https://github.com/euroargodev/sensor_metadata_json

    Such meta-data structures are expected to come from sensor manufacturer (web-api or file).

    OEM : Original Equipment Manufacturer

    Examples
    --------
    .. code-block:: python

        ArgoSensorMetaData()

        ArgoSensorMetaData(validate=True)  # Run json schema validation compliance when necessary

        ArgoSensorMetaData().from_dict(jsdata)  # Use any compliant json data

        ArgoSensorMetaData().from_rbr(208380)  # Direct call to the RBR api


    """
    _schema_src = "https://raw.githubusercontent.com/euroargodev/sensor_metadata_json/refs/heads/main/schemas/argo.sensor.schema.json"
    """URI of the argo sensor JSON schema"""

    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        validate: bool = False,
        **kwargs,
    ):
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

        self._run_validation = validate
        self.schema = self._read_schema()  # requires a self._fs instance

        self.sensor_info: Optional[SensorInfo] = None
        self.context: Optional[Context] = None
        self.sensors: List[Sensor] = field(default_factory=list)
        self.parameters: List[Parameter] = field(default_factory=list)
        self.instrument_vendorinfo: Optional[Dict[str, Any]] = None

        if json_data:
            self.from_dict(json_data)

    def _empty_str(self):
        summary = [f"<oemsensor>"]
        summary.append("This object has no sensor info. You can use one of the following methods:")
        for meth in [
            "from_rbr(serial_number)",
            "from_dict(dict_or_json_data)",
        ]:
            summary.append(f"  ╰┈➤ ArgoSensorMetaDataOem().{meth}")
        return summary

    def __repr__(self):
        if self.sensor_info:

            sensor_described = (
                self.sensor_info.sensor_described if self.sensor_info else "N/A"
            )
            created_by = self.sensor_info.created_by if self.sensor_info else "N/A"
            date_creation = self.sensor_info.date_creation if self.sensor_info else "N/A"
            sensor_count = len(self.sensors) if self.sensor_info else 0
            parameter_count = len(self.parameters) if self.sensor_info else 0

            summary = [f"<oemsensor.{sensor_described}>"]
            summary.append(f"created_by: '{created_by}'")
            summary.append(f"date_creation: '{date_creation}'")
            summary.append(f"sensors: {sensor_count} {[urnparser(s.SENSOR)['termid'] for s in self.sensors]}")
            summary.append(f"parameters: {parameter_count} {[urnparser(s.PARAMETER)['termid'] for s in self.parameters]}")
            summary.append(f"instrument_vendorinfo: {self.instrument_vendorinfo}")

        else:
            summary = self._empty_str()

        return "\n".join(summary)

    def _repr_html_(self):
        if self.sensor_info:
            return OemMetaDataDisplay(self).html
        else:
            return self._empty_str()

    def _ipython_display_(self):
        from IPython.display import display, HTML

        if self.sensor_info:
            display(HTML(OemMetaDataDisplay(self).html))
        else:
            display("\n".join(self._empty_str()))

    def _read_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        # todo: implement static asset backup to load schema
        schema = self._fs.open_json(self._schema_src)
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

    def to_json_file(self, file_path: str) -> None:
        """Save meta-data to a JSON file

        Notes
        -----
        The output json file is compliant with the Argo sensor meta-data JSON schema :attr:`ArgoSensorMetaDataOem.schema`
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def from_rbr(self, serial_number: str, **kwargs):
        """Fetch sensor metadata from RBR API and return an ArgoSensorMetaDataOem instance

        Parameters
        ----------
        serial_number : str
            Sensor serial number from RBR
        kwargs : dict
            Additional keyword arguments passed to the constructor of ArgoSensorMetaDataOem

        Notes
        -----
        The instance :class:`httpstore` is automatically updated to use the OPTIONS value for ``rbr_api_key``.
        """
        # Ensure that the instance httpstore has the appropriate authorization key:
        fss = self._fs.fs.fs if getattr(self._fs, 'cache') else self._fs.fs
        headers = fss.client_kwargs.get("headers", {})
        headers.update({"Authorization": kwargs.get("rbr_api_key", OPTIONS["rbr_api_key"])})
        fss._session = None  # Reset fsspec aiohttp.ClientSession

        uri = f"{OPTIONS['rbr_api']}/instruments/{serial_number}/argometadatajson"
        data = self._fs.open_json(uri)

        return self.from_dict(data)

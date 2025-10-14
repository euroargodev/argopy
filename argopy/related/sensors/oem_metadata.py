import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from  pathlib import Path
from zipfile import ZipFile
from referencing import Registry, Resource
import jsonschema
import logging
import warnings

from ...stores import httpstore, filestore
from ...options import OPTIONS
from ...utils import urnparser
from ...errors import InvalidDatasetStructure
from .oem_metadata_repr import OemMetaDataDisplay, ParameterDisplay


log = logging.getLogger("argopy.related.sensors")


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
    SENSOR_SERIAL_NO: str

    # FIRMWARE VERSION attributes are temporarily optional to handle the wrong key used by RBR
    # see https://github.com/euroargodev/sensor_metadata_json/issues/20
    SENSOR_FIRMWARE_VERSION: str = None  # RBR
    SENSOR_MODEL_FIRMWARE: str = None  # Correct schema key

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

    def _attr2str(self, x):
        """Return a class attribute, or 'n/a' if it's None, {} or ""."""
        value = getattr(self, x, None)
        if type(value) is str:
            return value if value and value.strip() else 'n/a'
        elif type(value) is dict:
            if len(value.keys()) == 0:
                return 'n/a'
            else:
                return value
        else:
            return value

    def __repr__(self):

        def key2str(d, x):
            """Return a dict value as a string, or 'n/a' if it's None or empty."""
            value = d.get(x, None)
            return value if value and value.strip() else 'n/a'

        summary = [f"<oemsensor.sensor.{self.SENSOR_SERIAL_NO}>"]
        summary.append(f"  SENSOR: {self.SENSOR} ({self.SENSOR_uri})")
        summary.append(f"  SENSOR_MAKER: {self.SENSOR_MAKER} ({self.SENSOR_MAKER_uri})")
        summary.append(f"  SENSOR_MODEL: {self.SENSOR_MODEL} ({self.SENSOR_MODEL_uri})")
        if getattr(self, "SENSOR_MODEL_FIRMWARE", None) is None:
            summary.append(f"  SENSOR_FIRMWARE_VERSION: {self._attr2str('SENSOR_FIRMWARE_VERSION')} (but should be 'SENSOR_MODEL_FIRMWARE') ")
        else:
            summary.append(f"  SENSOR_MODEL_FIRMWARE: {self._attr2str('SENSOR_MODEL_FIRMWARE')}")
        summary.append(f"  sensor_vendorinfo:")
        for key in self.sensor_vendorinfo.keys():
            summary.append(f"    - {key}: {key2str(self.sensor_vendorinfo, key)}")
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

    def _attr2str(self, x):
        """Return a class attribute, or 'n/a' if it's None, {} or ""."""
        value = getattr(self, x, None)
        if type(value) is str:
            return value if value and value.strip() else 'n/a'
        elif type(value) is dict:
            if len(value.keys()) == 0:
                return 'n/a'
            else:
                return value
        else:
            return value

    @property
    def _has_calibration_data(self):
        s = "".join([str(self._attr2str(key)) for key in
                     ['PREDEPLOYMENT_CALIB_EQUATION',
                      'PREDEPLOYMENT_CALIB_COEFFICIENT_LIST',
                      'PREDEPLOYMENT_CALIB_COMMENT',
                      'PREDEPLOYMENT_CALIB_DATE'] if self._attr2str(key) != 'n/a'])
        return len(s) > 0

    def __repr__(self):

        summary = [f"<oemsensor.parameter.{self.PARAMETER}>"]
        summary.append(f"  PARAMETER: {self.PARAMETER} ({self.PARAMETER_uri})")
        summary.append(f"  PARAMETER_SENSOR: {self.PARAMETER_SENSOR} ({self.PARAMETER_SENSOR_uri})")

        for key in ['UNITS', 'ACCURACY', 'RESOLUTION']:
            p = f"PARAMETER_{key}"
            summary.append(f"  {key}: {self._attr2str(p)}")

        summary.append(f"  PREDEPLOYMENT CALIBRATION:")
        for key in ['EQUATION', 'COEFFICIENT', 'COMMENT', 'DATE']:
            p = f"PREDEPLOYMENT_CALIB_{key}"
            summary.append(f"    - {key}: {self._attr2str(p)}")

        for key in ['parameter_vendorinfo', 'predeployment_vendorinfo']:
            summary.append(f"  {key}: {self._attr2str(p)}")
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
    _schema_root = "https://raw.githubusercontent.com/euroargodev/sensor_metadata_json/refs/heads/main/schemas"
    """URI root to argo JSON schema"""

    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        validate: bool = False,
        validation_error: Literal["warn", "raise", "ignore"] = "warn",
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
        self._validation_error = validation_error
        self.schema = self._read_schema()  # requires a self._fs instance

        self.sensor_info: Optional[SensorInfo] = None
        self.context: Optional[Context] = None
        self.sensors: List[Sensor] = field(default_factory=list)
        self.parameters: List[Parameter] = field(default_factory=list)
        self.instrument_vendorinfo: Optional[Dict[str, Any]] = None
        self._serial_number = None
        self._local_certificates = None

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
                self.sensor_info.sensor_described if self.sensor_info else "n/a"
            )
            created_by = self.sensor_info.created_by if self.sensor_info else "n/a"
            date_creation = self.sensor_info.date_creation if self.sensor_info else "n/a"
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

    def _read_schema(self, ref="argo.sensor.schema.json") -> Dict[str, Any]:
        """Load a JSON schema for validation."""
        # todo: implement static asset backup to load schema offline
        uri = f"{self._schema_root}/{ref}"
        schema = self._fs.open_json(uri)
        return schema

    def validate(self, data):
        """Validate meta-data against the Argo sensor json schema"""
        # Set a method to resolve references to subschemas
        registry = Registry(retrieve=lambda x: Resource.from_contents(self._read_schema(x)))

        # Select the validator based on $schema property in schema
        # (validators correspond to various drafts of JSON Schema)
        validator = jsonschema.validators.validator_for(self.schema)

        # Create the validator using the registry and associated resolver
        v = validator(self.schema, registry=registry)

        try:
            v.validate(data)
        except Exception as error:
            if self._validation_error == "raise":
                raise error
            elif self._validation_error == "warn":
                warnings.warn(str(error))
            else:
                log.error(error)

        # Create a list of errors, if any
        errors = list(v.evolve(schema=self.schema).iter_errors(v, data))
        return errors

    def from_dict(self, data: Dict[str, Any]):
        """Load data from a dictionary and possibly validate"""
        if self._run_validation:
            self.validate(data)

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
        """Fetch sensor metadata from RBR API

        We also download certificates if available

        Parameters
        ----------
        serial_number : str
            Sensor serial number from RBR

        Notes
        -----
        The instance :class:`httpstore` is automatically updated to use the OPTIONS value for ``rbr_api_key``.
        """
        self._serial_number = serial_number

        # Ensure that the instance httpstore has the appropriate authorization key:
        fss = self._fs.fs.fs if getattr(self._fs, 'cache') else self._fs.fs
        headers = fss.client_kwargs.get("headers", {})
        headers.update({"Authorization": kwargs.get("rbr_api_key", OPTIONS["rbr_api_key"])})
        fss._session = None  # Reset fsspec aiohttp.ClientSession

        uri = f"{OPTIONS['rbr_api']}/instruments/{self._serial_number}/argometadatajson"
        data = self._fs.open_json(uri)
        obj = self.from_dict(data)

        # Download RBR zip archive with calibration certificates in PDFs:
        obj = obj.certificates_rbr(action='download', quiet=True)

        return obj

    def certificates_rbr(self, action: Literal["download", "open"] = "download", **kwargs):
        """Download RBR zip archive with calibration certificates in PDFs

        Certificate PDF files are written to the OPTIONS['cachedir'] folder

        """
        cdir = Path(OPTIONS['cachedir']).joinpath("RBR_certificates")
        cdir.mkdir(parents=True, exist_ok=True)
        local_zip_path = cdir.joinpath(f"RBRcertificates_{self._serial_number}.zip")
        lfs = filestore()
        quiet = kwargs.get('quiet', False)

        # Check if we can continue:
        if self._serial_number is not None:
            new = False

            # Trigger download if necessary:
            if not lfs.exists(local_zip_path):
                new = True
                certif_uri = f"{OPTIONS['rbr_api']}/instruments/{self._serial_number}/certificates"
                with open(local_zip_path, 'wb') as local_zip:
                    with self._fs.open(certif_uri) as remote_zip:
                        local_zip.write(remote_zip.read())

                # Expand locally:
                with ZipFile(local_zip_path, "r") as local_zip:
                    local_zip.testzip()
                    local_zip.extractall(cdir)

            # List PDF certificates:
            with ZipFile(local_zip_path, "r") as local_zip:
                local_zip.testzip()
                info = local_zip.infolist()
            certificates = []
            for doc in info:
                certificates.append(Path(cdir).joinpath(doc.filename))
            self.local_certificates = certificates

            if not quiet:
                for f in self.local_certificates:
                    if new:
                        s = f"One RBR certificate file written to: {f}"
                    else:
                        s = f"One RBR certificate file already in: {f}"
                    print(s)
        else:
            raise InvalidDatasetStructure(f"You must load meta-data for a given RBR sensor serial number first. Use the 'from_rbr' method.")

        if action == 'download':
            return self
        elif action == 'open':
            subp = []
            for f in self.local_certificates:
                subp.append(lfs.open_subprocess(str(f)))
            if not quiet:
                return subp
        else:
            raise ValueError(f"Unknown action {action}")
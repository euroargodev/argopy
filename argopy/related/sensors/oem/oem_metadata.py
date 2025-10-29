from dataclasses import field
from typing import List, Dict, Optional, Any, Literal
from pathlib import Path
from zipfile import ZipFile

import json
import logging
import warnings
import pandas as pd
from html import escape

from ....stores import httpstore, filestore
from ....options import OPTIONS
from ....utils import urnparser, path2assets
from ....errors import InvalidDatasetStructure

from ..utils import has_jsonschema
from .oem_metadata_repr import OemMetaDataDisplay
from .accessories import SensorInfo, Context, Sensor, Parameter

if has_jsonschema:
    from referencing import Registry, Resource
    import jsonschema


log = logging.getLogger("argopy.related.sensors.oem")

SENSOR_JS_EXAMPLES = filestore().open_json(
    Path(path2assets).joinpath("sensor_metadata_examples.json")
)["data"]["uri"]


class OEMSensorMetaData:
    """OEM sensor meta-data

    A class helper to work with sensor meta-data complying to schema from https://github.com/euroargodev/sensor_metadata_json

    Such meta-data structures are expected to come from sensor manufacturer (web-api or file).

    .. note::
    
        OEM : Original Equipment Manufacturer

    Examples
    --------
    .. code-block:: python

        OEMSensorMetaData()

        OEMSensorMetaData(validate=True)  # Use this option to run json schema validation compliance when necessary

        OEMSensorMetaData().from_rbr(208380)  # Direct call to RBR api with a serial number

        OEMSensorMetaData().from_seabird(2444, 'SATLANTIC_OCR504_ICSW')  # Direct call to Seabird api with a serial number and model name

        OEMSensorMetaData().list_examples

        OEMSensorMetaData().from_examples('WETLABS-ECO_FLBBAP2-8589')

        OEMSensorMetaData().from_dict(jsdata)  # Use any compliant json data

    """

    _schema_root = "https://raw.githubusercontent.com/euroargodev/sensor_metadata_json/refs/heads/main/schemas"
    """URI root to argo JSON schema"""

    def __init__(
        self,
        json_data: Optional[Dict[str, Any]] = None,
        validate: bool = True,
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

        if has_jsonschema:
            self._run_validation = validate
        else:
            warnings.warn(f"Cannot run JSON validation without the 'jsonschema' library. Please install it manually. Fall back on setting `validate=False`. ")
            self._run_validation = False

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

    def _repr_hint(self):
        summary = [f"<oemsensor>"]
        summary.append(
            "This object has no sensor info. You can use one of the following methods:"
        )
        for meth in [
            "from_rbr(serial_number)",
            "from_seabird(serial_number, model_name)",
            "from_dict(dict_or_json_data)",
        ]:
            summary.append(f"  ╰┈➤ OEMSensorMetaData().{meth}")
        return summary

    def __repr__(self):
        if self.sensor_info:

            sensor_described = self.sensor_info._attr2str('sensor_described')
            created_by = self.sensor_info._attr2str('created_by')
            date_creation = self.sensor_info._attr2str('date_creation')
            link = self.sensor_info._attr2str('link')

            sensor_count = len(self.sensors)
            parameter_count = len(self.parameters)

            summary = [f"<oemsensor><{sensor_described}>"]
            summary.append(f"created_by: '{created_by}'")
            summary.append(f"date_creation: '{date_creation}'")
            summary.append(f"link: '{link}'")
            summary.append(
                f"sensors: {sensor_count} {[urnparser(s.SENSOR)['termid'] for s in self.sensors]}"
            )
            summary.append(
                f"parameters: {parameter_count} {[urnparser(s.PARAMETER)['termid'] for s in self.parameters]}"
            )
            summary.append(f"instrument_vendorinfo: {self.instrument_vendorinfo}")

        else:
            summary = self._repr_hint()

        return "\n".join(summary)

    def _repr_html_(self):
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return OemMetaDataDisplay(self).html

    # def _ipython_display_(self):
    #     from IPython.display import display, HTML
    #
    #     if self.sensor_info:
    #         display(HTML(OemMetaDataDisplay(self).html))
    #     else:
    #         display("\n".join(self._repr_hint()))

    def _read_schema(self, ref="argo.sensor.schema.json") -> Dict[str, Any]:
        """Load a JSON schema for validation

        Fall back on static version if online resource not available
        """
        uri = f"{self._schema_root}/{ref}"
        try:
            schema = self._fs.open_json(uri)
        except:
            # Fall back on static assets version
            local_uri = Path(path2assets).joinpath("schema").joinpath(ref)
            fs = filestore()
            updated = pd.Timestamp(fs.info(local_uri)["mtime"], unit="s")
            warnings.warn(
                f"\nCan't get '{ref}' schema from the official online resource ({uri}).\nFall back on a static version packaged with this release at {updated}."
            )
            schema = fs.open_json(local_uri)

        return schema

    def validate(self, data):
        """Validate meta-data against the Argo sensor json schema"""
        # Set a method to resolve references to subschemas
        registry = Registry(
            retrieve=lambda x: Resource.from_contents(self._read_schema(x))
        )

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
                warnings.warn(f"\nJSON schema validation error: {str(error)}")
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
        if data.get("@context", None) is not None:
            self.context = Context(
                **{
                    k.replace("::", "").replace(":", "_"): v
                    for k, v in data["@context"].items()
                }
            )
        else:
            self.context = Context()
        self.sensors = [Sensor(**sensor) for sensor in data["SENSORS"]]
        self.parameters = [Parameter(**param) for param in data["PARAMETERS"]]
        self.instrument_vendorinfo = data.get("instrument_vendorinfo", None)

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
        """Save meta-data to a JSON file - in dev.

        Notes
        -----
        The output json file should be compliant with the Argo sensor meta-data JSON schema :attr:`OEMSensorMetaData.schema`
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def from_rbr(self, serial_number: str, **kwargs) -> 'OEMSensorMetaData':
        """Fetch sensor metadata from "RBRargo Product Lookup" web-API

        We also download certificates if available

        TODO: Check mark if the sensor is ok with dynamic correction or not

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
        fss = self._fs.fs.fs if getattr(self._fs, "cache") else self._fs.fs
        headers = fss.client_kwargs.get("headers", {})
        headers.update(
            {"Authorization": kwargs.get("rbr_api_key", OPTIONS["rbr_api_key"])}
        )
        fss._session = None  # Reset fsspec aiohttp.ClientSession

        uri = f"{OPTIONS['rbr_api']}/instruments/{self._serial_number}/argometadatajson"
        data = self._fs.open_json(uri)
        obj = self.from_dict(data)

        # Also download RBR zip archive with calibration certificates in PDFs:
        obj = obj._certificates_rbr(action="download", quiet=True)

        # Finally reset httpstore parameters:
        headers = fss.client_kwargs.get("headers")
        headers.pop("Authorization", None)
        fss._session = None  # Reset fsspec aiohttp.ClientSession

        return obj

    def _certificates_rbr(
        self, action: Literal["download", "open"] = "download", **kwargs
    ):
        """Download RBR zip archive with calibration certificates in PDFs

        Certificate PDF files are written to the OPTIONS['cachedir'] folder

        Notes
        -----
        We keep this method private because it is expected to be called only by the self.from_rbr() method.
        This ensures that the httpstore has the appropriate authorization key.

        """
        cdir = Path(OPTIONS["cachedir"]).joinpath("RBR_certificates")
        cdir.mkdir(parents=True, exist_ok=True)
        local_zip_path = cdir.joinpath(f"RBRcertificates_{self._serial_number}.zip")
        lfs = filestore()
        quiet = kwargs.get("quiet", False)

        # Check if we can continue:
        if self._serial_number is not None:
            new = False

            # Trigger download if necessary:
            if not lfs.exists(local_zip_path):
                new = True
                certif_uri = f"{OPTIONS['rbr_api']}/instruments/{self._serial_number}/certificates"
                with open(local_zip_path, "wb") as local_zip:
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
            raise InvalidDatasetStructure(
                f"You must load meta-data for a given RBR sensor serial number first. Use the 'from_rbr' method."
            )

        if action == "download":
            return self
        elif action == "open":
            subp = []
            for f in self.local_certificates:
                subp.append(lfs.open_subprocess(str(f)))
            if not quiet:
                return subp
        else:
            raise ValueError(f"Unknown action {action}")

    def from_seabird(self, serial_number: str, sensor_model: str, **kwargs) -> 'OEMSensorMetaData':
        """Fetch sensor metadata from Seabird-Scientific "Instrument Metadata Portal" web-API

        Parameters
        ----------
        serial_number : str
            Sensor serial number from RBR
        """
        # The Seabird api requires a sensor model (R27) with a serial number.
        # This could easily be done if we get the s/n by searching float metadata.
        # In other word, it's easy to go from a sensor_model to a serial_number.
        # But it is much more complicated to go from a serial_number to a sensor_model.
        # so, for the time being, we'll ask users to specify a sensor_model.

        url = f"{OPTIONS['seabird_api']}?SENSOR_SERIAL_NO={serial_number}&SENSOR_MODEL={sensor_model}"
        data = self._fs.open_json(url)

        # Temporary fix for errors in SBE api output:
        data.update({'sensor_info': data['json_info']})
        data['sensor_info'].update({'link': data['sensor_info']['created_by']})
        data['sensor_info'].update({'created_by': 'Sea-Bird Instrument Metadata Portal'})
        data['sensor_info'].update({'sensor_described': f"{sensor_model}-{serial_number}"})
        data.pop('json_info', None)

        # Check in vendor info for missing mandatory parameter attributes:
        for ii, param in enumerate(data['PARAMETERS']):
            vendorinfo = param['parameter_vendorinfo']
            if param.get('PARAMETER_UNITS', None) is None:
                if 'units' in vendorinfo:
                    data['PARAMETERS'][ii]['PARAMETER_UNITS'] = vendorinfo['units']
            if param.get('PREDEPLOYMENT_CALIB_COMMENT', None) is None:
                if 'units' in vendorinfo:
                    data['PARAMETERS'][ii]['PREDEPLOYMENT_CALIB_COMMENT'] = vendorinfo['comments']
            if param.get('PREDEPLOYMENT_CALIB_DATE', None) is None:
                if 'units' in vendorinfo:
                    data['PARAMETERS'][ii]['PREDEPLOYMENT_CALIB_DATE'] = vendorinfo['calibration_date']

        # Create and return an instance
        obj = self.from_dict(data)
        return obj

    @property
    def list_examples(self):
        """List of example names"""
        return [k for k in SENSOR_JS_EXAMPLES.keys()]

    def from_examples(self, eg: str = None, **kwargs):
        if eg not in self.list_examples:
            raise ValueError(
                f"Unknown sensor example: '{eg}'. \n Use one in: {self.list_examples}"
            )
        data = self._fs.open_json(SENSOR_JS_EXAMPLES[eg])
        return self.from_dict(data)

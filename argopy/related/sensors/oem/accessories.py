from dataclasses import dataclass
from typing import Dict, Optional, Any
from html import escape

from ....options import OPTIONS
from ....utils import urnparser

from .oem_metadata_repr import ParameterDisplay


@dataclass
class SensorInfo:
    created_by: str
    date_creation: str  # ISO 8601 datetime string
    link: str
    format_version: str
    contents: str

    # Made optional to accommodate errors in OEM data
    sensor_described: str = None

    def _attr2str(self, x):
        """Return a class attribute, or 'n/a' if it's None or ""."""
        value = getattr(self, x, None)
        if value is None:
            return "n/a"
        elif type(value) is str:
            return value if value and value.strip() else "n/a"
        else:
            return value


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
        if value is None:
            return "n/a"
        elif type(value) is str:
            return value if value and value.strip() else "n/a"
        elif type(value) is dict:
            if len(value.keys()) == 0:
                return "n/a"
            else:
                return value
        else:
            return value

    def __repr__(self):

        def key2str(d, x):
            """Return a dict value as a string, or 'n/a' if it's None or empty."""
            value = d.get(x, None)
            return value if value and value.strip() else "n/a"

        summary = [f"<oemsensor.sensor><{self.SENSOR}><{self.SENSOR_SERIAL_NO}>"]
        summary.append(f"  SENSOR: {self.SENSOR} ({self.SENSOR_uri})")
        summary.append(f"  SENSOR_MAKER: {self.SENSOR_MAKER} ({self.SENSOR_MAKER_uri})")
        summary.append(f"  SENSOR_MODEL: {self.SENSOR_MODEL} ({self.SENSOR_MODEL_uri})")
        if getattr(self, "SENSOR_MODEL_FIRMWARE", None) is None:
            summary.append(
                f"  SENSOR_FIRMWARE_VERSION: {self._attr2str('SENSOR_FIRMWARE_VERSION')} (but should be 'SENSOR_MODEL_FIRMWARE') "
            )
        else:
            summary.append(
                f"  SENSOR_MODEL_FIRMWARE: {self._attr2str('SENSOR_MODEL_FIRMWARE')}"
            )
        if getattr(self, "sensor_vendorinfo", None) is not None:
            summary.append(f"  sensor_vendorinfo:")
            for key in self.sensor_vendorinfo.keys():
                summary.append(f"    - {key}: {key2str(self.sensor_vendorinfo, key)}")
        else:
            summary.append(f"  sensor_vendorinfo: None")
        return "\n".join(summary)


@dataclass
class Parameter:
    PARAMETER: str  # SDN:R03::PRES
    PARAMETER_SENSOR: str  # SDN:R25::CTD_PRES
    PARAMETER_ACCURACY: str
    PARAMETER_RESOLUTION: str
    PREDEPLOYMENT_CALIB_COEFFICIENT_LIST: Dict[str, str]

    # Made optional to accommodate errors in OEM data
    PARAMETER_UNITS: str = None
    PREDEPLOYMENT_CALIB_EQUATION: str = None
    PREDEPLOYMENT_CALIB_COMMENT: str = None
    PREDEPLOYMENT_CALIB_DATE: str = None

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
        if value is None:
            return "n/a"
        elif type(value) is str:
            return value if value and value.strip() else "n/a"
        elif type(value) is dict:
            if len(value.keys()) == 0:
                return "n/a"
            else:
                return value
        else:
            return value

    @property
    def _has_calibration_data(self):
        s = "".join(
            [
                str(self._attr2str(key))
                for key in [
                    "PREDEPLOYMENT_CALIB_EQUATION",
                    "PREDEPLOYMENT_CALIB_COEFFICIENT_LIST",
                    "PREDEPLOYMENT_CALIB_COMMENT",
                    "PREDEPLOYMENT_CALIB_DATE",
                ]
                if self._attr2str(key) != "n/a"
            ]
        )
        return len(s) > 0

    def __repr__(self):

        summary = [f"<oemsensor.parameter><{self.PARAMETER}>"]
        summary.append(f"  PARAMETER: {self.PARAMETER} ({self.PARAMETER_uri})")
        summary.append(
            f"  PARAMETER_SENSOR: {self.PARAMETER_SENSOR} ({self.PARAMETER_SENSOR_uri})"
        )

        for key in ["UNITS", "ACCURACY", "RESOLUTION"]:
            p = f"PARAMETER_{key}"
            summary.append(f"  {key}: {self._attr2str(p)}")

        summary.append(f"  PREDEPLOYMENT CALIBRATION:")
        for key in ["EQUATION", "COEFFICIENT", "COMMENT", "DATE"]:
            p = f"PREDEPLOYMENT_CALIB_{key}"
            summary.append(f"    - {key}: {self._attr2str(p)}")

        for key in ["parameter_vendorinfo", "predeployment_vendorinfo"]:
            if getattr(self, key, None) is not None:
                summary.append(f"  {key}: {self._attr2str(key)}")
            else:
                summary.append(f"  {key}: None")
        return "\n".join(summary)

    def _repr_html_(self):
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return ParameterDisplay(self).html

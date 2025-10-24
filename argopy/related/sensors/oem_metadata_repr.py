from functools import lru_cache
import importlib
from numpy.random import randint

try:
    from importlib.resources import files  # New in version 3.9
except ImportError:
    from pathlib import Path

    files = lambda x: Path(  # noqa: E731
        importlib.util.find_spec(x).submodule_search_locations[0]
    )

from ...utils import urnparser


STATIC_FILES = (
    ("argopy.static.css", "argopy.css"),
    ("argopy.static.css", "oemsensor.css"),
)

@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [
        files(package).joinpath(resource).read_text(encoding="utf-8")
        for package, resource in STATIC_FILES
    ]


def urn_html(this_urn):
    x = urnparser(this_urn)
    if x.get('version') != "":
        return f"<strong>{x.get('termid', '?')}</strong> ({x.get('listid', '?')}, {x.get('version', 'n/a')})"
    else:
        return f"<strong>{x.get('termid', '?')}</strong> ({x.get('listid', '?')})"



class OemMetaDataDisplay:

    def __init__(self, obj):
        self.OEMsensor = obj

    @property
    def css_style(self):
        return "\n".join(_load_static_files())

    @property
    def html(self):
        # Generate a dummy random id to be used in html elements for this output only
        # This avoids messing up javascript actions between notebook cells
        uid = f"{randint(low=1e7):8d}".strip()

        # --- Header ---
        header_html = f"""
        <h1 class='oemsensor'>Argo Sensor Metadata: {self.OEMsensor.sensor_info._attr2str('sensor_described')}</h1>
        <p class='oemsensor'><strong>Created by:</strong> {self.OEMsensor.sensor_info._attr2str('created_by')} |
           <strong>Date:</strong> {self.OEMsensor.sensor_info._attr2str('date_creation')}</p>
        """

        # --- Sensors ---
        sensors_html = """
        <h2 class='oemsensor'>List of sensors:</h2>
        <table class='oemsensor'>
            <thead>
                <tr class='oemsensor'>
                    <th class='oemsensor'>Sensor</th>
                    <th class='oemsensor'>Maker</th>
                    <th class='oemsensor'>Model</th>
                    <th class='oemsensor'>Firmware</th>
                    <th class='oemsensor'>Serial No</th>
                </tr>
            </thead>
            <tbody>
        """

        for ii, sensor in enumerate(self.OEMsensor.sensors):

            if getattr(sensor, "SENSOR_MODEL_FIRMWARE", None) is None:
                firmware = f"{sensor._attr2str('SENSOR_FIRMWARE_VERSION')}"
            else:
                firmware = f"{sensor._attr2str('SENSOR_MODEL_FIRMWARE')}"

            sensors_html += f"""
                <tr class='oemsensor'>
                    <td class='oemsensor'><a href='{sensor.SENSOR_uri}'>{urn_html(sensor.SENSOR)}</a></td>
                    <td class='oemsensor'><a href='{sensor.SENSOR_MAKER_uri}'>{urn_html(sensor.SENSOR_MAKER)}</a></td>
                    <td class='oemsensor'><a href='{sensor.SENSOR_MODEL_uri}'>{urn_html(sensor.SENSOR_MODEL)}</a></td>
                    <td class='oemsensor'>{firmware}</td>
                    <td class='oemsensor'>{sensor._attr2str('SENSOR_SERIAL_NO')}</td>
                </tr>
            """

        sensors_html += "</tbody></table>"

        # --- Parameters ---
        parameters_html = """
        <h2 class='oemsensor'>List of parameters:</h2>
        <table class='oemsensor'>
            <thead>
                <tr class='oemsensor'>
                    <th class='oemsensor'>Parameter</th>
                    <th class='oemsensor'>Sensor</th>
                    <th class='oemsensor'>Units</th>
                    <th class='oemsensor'>Accuracy</th>
                    <th class='oemsensor'>Resolution</th>
                    <th class='oemsensor'>Calibration details</th>
                </tr>
            </thead>
            <tbody>
        """

        for ip, param in enumerate(self.OEMsensor.parameters):
            if getattr(param, 'PREDEPLOYMENT_CALIB_EQUATION', None) is not None:
                PREDEPLOYMENT_CALIB_EQUATION = param._attr2str('PREDEPLOYMENT_CALIB_EQUATION').split(';')
                PREDEPLOYMENT_CALIB_EQUATION = [p.replace("=", " = ") for p in PREDEPLOYMENT_CALIB_EQUATION]
                PREDEPLOYMENT_CALIB_EQUATION = "<br>\t".join(PREDEPLOYMENT_CALIB_EQUATION)
            else:
                PREDEPLOYMENT_CALIB_EQUATION = "<i>This information is missing, but it should not !</i>"

            if getattr(param, 'PREDEPLOYMENT_CALIB_COEFFICIENT_LIST', None) is not None:
                PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = param._attr2str('PREDEPLOYMENT_CALIB_COEFFICIENT_LIST')
                s = []
                if isinstance(PREDEPLOYMENT_CALIB_COEFFICIENT_LIST, dict):
                    for key, value in PREDEPLOYMENT_CALIB_COEFFICIENT_LIST.items():
                        s.append(f"{key} = {value}")
                    PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = "<br>\t".join(s)
            else:
                PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = "<i>This information is missing, but it should not !</i>"


            details_html = f"""
            <tr class="parameter-details oemsensor" id="details-{uid}-{ip}">
                <td colspan="6">
                    <div class="details-content oemsensor">
                        <p class='oemsensor'><strong>Calibration Equation:</strong><br>{PREDEPLOYMENT_CALIB_EQUATION}</p>
                        <p class='oemsensor'><strong>Calibration Coefficients:</strong><br>{PREDEPLOYMENT_CALIB_COEFFICIENT_LIST}</p>
                        <p class='oemsensor'><strong>Calibration Comment:</strong><br>{param._attr2str('PREDEPLOYMENT_CALIB_COMMENT')}</p>
                        <p class='oemsensor'><strong>Calibration Date:</strong><br>{param._attr2str('PREDEPLOYMENT_CALIB_DATE')}</p>
                    </div>
                </td>
            </tr>
            """

            # param.PREDEPLOYMENT_CALIB_EQUATION
            if param._has_calibration_data:
                line = '<td class="oemsensor calibration" onclick="toggleDetails(\'details-%s-%i\')">Click for more</td>' % (uid, ip)
            else:
                line = f"<td class='oemsensor calibration'>n/a</td>"

            parameters_html += f"""
            <tr class="parameter-row oemsensor">
                <td class='oemsensor'><a href='{param.PARAMETER_uri}'>{urn_html(param.PARAMETER)}</a></td>
                <td class='oemsensor'><a href='{param.PARAMETER_SENSOR_uri}'>{urn_html(param.PARAMETER_SENSOR)}</a></td>
                <td class='oemsensor'>{param._attr2str('PARAMETER_UNITS')}</td>
                <td class='oemsensor'>{param._attr2str('PARAMETER_ACCURACY')}</td>
                <td class='oemsensor'>{param._attr2str('PARAMETER_RESOLUTION')}</td>
                {line}
            </tr>
            {details_html}
            """

        parameters_html += "</tbody></table>"

        # --- JavaScript for Toggle ---
        js = """
        <script>
            function toggleDetails(id) {
                const details = document.getElementById(id);
                if (details.style.display === "none" || details.style.display === "") {
                    details.style.display = "table-row";
                } else {
                    details.style.display = "none";
                }
            }
        </script>
        """

        # --- Vendor Info ---
        vendor_html = ""
        if self.OEMsensor.instrument_vendorinfo:
            vendor_html = f"""
            <h2 class='oemsensor'>Instrument Vendor Info:</h2>
            <pre class='oemsensor'>{self.OEMsensor.instrument_vendorinfo}</pre>
            """

        # --- Combine All HTML ---
        full_html = f"""
        <style>{self.css_style}</style>\n
        {header_html}\n
        {sensors_html}\n
        {parameters_html}\n
        {js}\n
        {vendor_html}
        """
        return full_html

    def _repr_html_(self):
        return self.html


class ParameterDisplay:

    def __init__(self, obj):
        self.data = obj

    @property
    def css_style(self):
        return "\n".join(_load_static_files())

    @property
    def html(self):
        param = self.data

        # --- Header ---
        header_html = f"<h1 class='oemsensor'>Argo Sensor Metadata for Parameter: <a href='{param.PARAMETER_uri}'>{urn_html(param.PARAMETER)}</a></h1>"

        if param.parameter_vendorinfo is not None:
            info = " | ".join([f"<strong>{p}</strong> {v}" for p, v in param.parameter_vendorinfo.items()])
            header_html += f"<p class='oemsensor'>{info}</p>"

        # --- Parameter details ---
        html = """
        <!--<h2 class='oemsensor'>List of parameters:</h2>-->
        <table class='oemsensor'>
            <thead>
                <tr class='oemsensor'>
                    <th class='oemsensor'>Sensor</th>
                    <th class='oemsensor'>Units</th>
                    <th class='oemsensor'>Accuracy</th>
                    <th class='oemsensor'>Resolution</th>
                </tr>
            </thead>
            <tbody>
        """
        html += f"""
        <tr class="parameter-row oemsensor">
            <td class='oemsensor'><a href='{param.PARAMETER_SENSOR_uri}'>{urn_html(param.PARAMETER_SENSOR)}</a></td>
            <td class='oemsensor'>{param._attr2str('PARAMETER_UNITS')}</td>
            <td class='oemsensor'>{param._attr2str('PARAMETER_ACCURACY')}</td>
            <td class='oemsensor'>{param._attr2str('PARAMETER_RESOLUTION')}</td>
        </tr>
        """

        PREDEPLOYMENT_CALIB_EQUATION = param._attr2str('PREDEPLOYMENT_CALIB_EQUATION').split(';')
        PREDEPLOYMENT_CALIB_EQUATION = [p.replace("=", " = ") for p in PREDEPLOYMENT_CALIB_EQUATION]
        PREDEPLOYMENT_CALIB_EQUATION = "<br>\t".join(PREDEPLOYMENT_CALIB_EQUATION)

        PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = param._attr2str('PREDEPLOYMENT_CALIB_COEFFICIENT_LIST')
        s = []
        if isinstance(PREDEPLOYMENT_CALIB_COEFFICIENT_LIST, dict):
            for key, value in PREDEPLOYMENT_CALIB_COEFFICIENT_LIST.items():
                s.append(f"{key} = {value}")
            PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = "<br>\t".join(s)

        html += f"""
        <tr class="parameter-row oemsensor">
            <td colspan="4">
                <div class="calibration-content oemsensor">
                    <p class='oemsensor'><strong>Calibration Equation:</strong><br>{PREDEPLOYMENT_CALIB_EQUATION}</p>
                    <p class='oemsensor'><strong>Calibration Coefficients:</strong><br>{PREDEPLOYMENT_CALIB_COEFFICIENT_LIST}</p>
                    <p class='oemsensor'><strong>Calibration Comment:</strong><br>{param._attr2str('PREDEPLOYMENT_CALIB_COMMENT')}</p>
                    <p class='oemsensor'><strong>Calibration Date:</strong><br>{param._attr2str('PREDEPLOYMENT_CALIB_DATE')}</p>
                </div>
            </td>
        </tr>
        """
        html += "</tbody></table>"

        # --- Vendor Info ---
        vendor_html = ""
        if param.predeployment_vendorinfo:
            vendor_html = f"""
            <h2 class='oemsensor'>Pre-deployment Vendor Info:</h2>
            <pre class='oemsensor'>{param.predeployment_vendorinfo}</pre>
            """

        # --- Combine All HTML ---
        full_html = f"""
        <style>{self.css_style}</style>\n
        {header_html}\n
        {html}\n
        {vendor_html}
        """
        return full_html

    def _repr_html_(self):
        return self.html
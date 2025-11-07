from functools import lru_cache
import importlib


try:
    from importlib.resources import files  # New in version 3.9
except ImportError:
    from pathlib import Path

    files = lambda x: Path(  # noqa: E731
        importlib.util.find_spec(x).submodule_search_locations[0]
    )

from argopy.utils import urnparser


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
    if x.get("version") != "":
        return f"<strong>{x.get('termid', '?')}</strong> ({x.get('listid', '?')}, {x.get('version', 'n/a')})"
    else:
        return f"<strong>{x.get('termid', '?')}</strong> ({x.get('listid', '?')})"


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
            info = " | ".join(
                [
                    f"<strong>{p}</strong> {v}"
                    for p, v in param.parameter_vendorinfo.items()
                ]
            )
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

        PREDEPLOYMENT_CALIB_EQUATION = param._attr2str(
            "PREDEPLOYMENT_CALIB_EQUATION"
        ).split(";")
        PREDEPLOYMENT_CALIB_EQUATION = [
            p.replace("=", " = ") for p in PREDEPLOYMENT_CALIB_EQUATION
        ]
        PREDEPLOYMENT_CALIB_EQUATION = "<br>\t".join(PREDEPLOYMENT_CALIB_EQUATION)

        PREDEPLOYMENT_CALIB_COEFFICIENT_LIST = param._attr2str(
            "PREDEPLOYMENT_CALIB_COEFFICIENT_LIST"
        )
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

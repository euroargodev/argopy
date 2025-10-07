import urllib
import importlib
import time
import threading
import logging

from .lists import list_available_data_src
from .checkers import isAPIconnected

try:
    importlib.import_module("matplotlib")  # noqa: E402
    from matplotlib.colors import to_hex
except ImportError:
    pass


log = logging.getLogger("argopy.utils.monitors")


def badge(label="label", message="message", color="green", insert=False):
    """Return or insert shield.io badge image

        Use the shields.io service to create a badge image

        https://img.shields.io/static/v1?label=<LABEL>&message=<MESSAGE>&color=<COLOR>

    Parameters
    ----------
    label: str
        Left side badge text
    message: str
        Right side badge text
    color: str
        Right side background color
    insert: bool
        Return url to badge image (False, default) or directly insert the image with HTML (True)

    Returns
    -------
    str or IPython.display.Image
    """
    from IPython.display import Image

    url = (
        "https://img.shields.io/static/v1?style=flat-square&label={}&message={}&color={}"
    ).format
    img = url(urllib.parse.quote(label), urllib.parse.quote(message), color)
    if not insert:
        return img
    else:
        return Image(url=img)


class fetch_status:
    """Fetch and report web API status"""

    def fetch(self):
        results = {}
        list_src = list_available_data_src()
        for api, mod in list_src.items():
            if getattr(mod, "api_server_check", None):
                status = isAPIconnected(api)
                message = "ok" if status else "offline"
                results[api] = {"value": status, "message": message}
        return results

    @property
    def text(self):
        results = self.fetch()
        rows = []
        for api in sorted(results.keys()):
            rows.append("src %s is: %s" % (api, results[api]["message"]))
        txt = " | ".join(rows)
        return txt

    def __repr__(self):
        return self.text

    @property
    def html(self):
        results = self.fetch()

        fs = 12

        def td_msg(bgcolor, txtcolor, txt):
            style = "background-color:%s;" % to_hex(bgcolor, keep_alpha=True)
            style += "border-width:0px;"
            style += "padding: 2px 5px 2px 5px;"
            style += "text-align:left;"
            style += "color:%s" % to_hex(txtcolor, keep_alpha=True)
            return "<td style='%s'>%s</td>" % (style, str(txt))

        td_empty = "<td style='border-width:0px;padding: 2px 5px 2px 5px;text-align:left'>&nbsp;</td>"

        html = []
        html.append(
            "<table style='border-collapse:collapse;border-spacing:0;font-size:%ipx'>"
            % fs
        )
        html.append("<tbody><tr>")
        cols = []
        for api in sorted(results.keys()):
            color = "yellowgreen" if results[api]["value"] else "darkorange"
            cols.append(td_msg("dimgray", "w", "src %s is" % api))
            cols.append(td_msg(color, "w", results[api]["message"]))
            cols.append(td_empty)
        html.append("\n".join(cols))
        html.append("</tr></tbody>")
        html.append("</table>")
        html = "\n".join(html)
        return html

    def _repr_html_(self):
        return self.html


class monitor_status:
    """Monitor data source status with a refresh rate"""

    def __init__(self, refresh=60):
        self.refresh_rate = refresh

        if self.runner == "notebook":
            import ipywidgets as widgets

            self.text = widgets.HTML(
                value=self.content,
                placeholder="",
                description="",
            )
            self.start()

    def __repr__(self):
        if self.runner != "notebook":
            return self.content
        else:
            return ""

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return "notebook"  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return "terminal"  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return "standard"  # Probably standard Python interpreter

    @property
    def content(self):
        if self.runner == "notebook":
            return fetch_status().html
        else:
            return fetch_status().text

    def work(self):
        while True:
            time.sleep(self.refresh_rate)
            self.text.value = self.content

    def start(self):
        from IPython.display import display

        thread = threading.Thread(target=self.work)
        display(self.text)
        thread.start()

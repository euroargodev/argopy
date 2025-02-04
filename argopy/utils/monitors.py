import json
import urllib
import importlib
import time
import threading
import pandas as pd
import copy
from typing import Union, List, Dict, Literal
import logging


log = logging.getLogger("argopy.utils.monitors")


try:
    importlib.import_module("matplotlib")  # noqa: E402
    from matplotlib.colors import to_hex
except ImportError:
    pass

from .lists import list_available_data_src
from .checkers import isAPIconnected


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


class ArgopyCarbon:
    """Compute argopy carbon footprint since last release

    Use the Green-Coding Solutions API to retrieve energy consumption data.

    Combined with the github API, this class aims to provide an easy method to retrieve the
    CI activities of argopy.

    Examples
    --------
    .. coding::python

        ArgopyCarbon().workflows
        ArgopyCarbon().measurements(branch='master', start_date='2024-01-01')
        ArgopyCarbon().measurements(branch='385/merge', start_date='2024-01-01')
        ArgopyCarbon().total_measurements(branches=['master', '385/merge'])

        ArgopyCarbon().releases
        ArgopyCarbon().lastreleasedate
        ArgopyCarbon().lastPRs
        ArgopyCarbon().get_PRtitle(385)
        ArgopyCarbon().get_PRmerged_since('2025-01-01')

        ArgopyCarbon().footprint_since_last_release()

    """

    owner = "euroargodev"
    repo = "argopy"
    workflows = [
        {"ID": "22344160", "Name": "CI tests"},
        {"ID": "25052179", "Name": "CI tests upstream"},
    ]

    def __init__(self):
        from ..stores import httpstore

        self.fs = httpstore(cache=True)

    def measurements(
        self,
        branch: str = "master",
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None,
        errors: Literal["raise", "ignore", "silent"] = "ignore",

    ) -> List[Dict]:
        """Return measurements for a given branch and date period

        Parameters
        ----------
        branch : str, default='master'
            Name of the branch to retrieve measurements for
        start_date : str, :class:`pd.Timestamp`, default=None
            Measurements starting date, default to 1 year before today
        end_date : str, :class:`pd.Timestamp`, default=None
            Measurements ending date, default to today
        errors: Literal, default: ``ignore``
            Define how to handle errors raised during data fetching:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

        Returns
        -------
        List[Dict] with workflows name, ID and measurements as :class:`pandas.DataFrame`
        """
        if end_date is None:
            end_date = pd.to_datetime("now", utc=True)
        else:
            end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = end_date - pd.Timedelta(365, unit="D")
        else:
            start_date = pd.to_datetime(start_date)

        long_names = [  # noqa: F841
            "Energy Value [J]",
            "Run ID",
            "Ran At",
            "Label",
            "CPU",
            "Commit Hash",
            "Duration [ms]",
            "Platform",
            "Avg. CPU Utilization",
            "Workflow",
            "?",
            "??",
            "???",
            "Grid Intensity [gCOE2/kWh]",
            "gCO2eq of run []",
        ]

        names = [
            "Energy",
            "runID",
            "Date",
            "Workflow_Step",
            "CPU_type",
            "CommitHash",
            "Duration",
            "platform",
            "Avg. CPU_utilization",
            "Workflow",
            "?",
            "??",
            "???",
            "GridIntensity",
            "gCO2eq",
        ]

        results = copy.deepcopy(self.workflows)
        for irow, workflow in enumerate(self.workflows):

            payload = {
                "repo": f"{self.owner}/{self.repo}",
                "branch": branch,
                "workflow": workflow["ID"],
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }

            uri = (
                "https://api.green-coding.io/v1/ci/measurements?"
                + urllib.parse.urlencode(payload)
            )

            try:
                data = self.fs.open_json(uri)

                df = pd.DataFrame(data["data"], columns=names)
                df["Duration"] = df["Duration"] / 1e6  # seconds
                df["gCO2eq"] = df["gCO2eq"] / 1e6  # gCO2eq
                df["Energy"] = df["Energy"] / 1e6  # Joules

                df2 = df[["Workflow_Step", "Energy", "Duration", "gCO2eq"]]
                df2 = df2.groupby("Workflow_Step").sum()

                results[irow]["measurements"] = df2

            except json.JSONDecodeError:
                msg = "No data returned, probably because the branch '%s' was not found at %s" % (branch, uri)
                if errors == 'raise':
                    from ..errors import DataNotFound

                    raise DataNotFound(msg)
                elif errors == 'ignore':
                    log.debug(msg)

            except Exception as e:
                if errors == 'raise':
                    raise
                elif errors == 'ignore':
                    log.error("Error: {e}")

        return results

    def total_measurements(self, branches: List[str], **kwargs) -> float:
        """Compute the cumulated measurements of gCO2eq for a list of branches

        Parameters
        ----------
        branches : List[str]
            List of branches to retrieve measurements for.
            For a given merged PR number, the branche name is '<PR>/merged'.
        **kwargs:
            Other Parameters are passed to :class:`ArgopyCarbon.measurements`

        Returns
        -------
        float
        """
        gCO2eq = 0
        for branch in branches:
            for wkf in self.measurements(branch=branch, **kwargs):
                if "measurements" in wkf:
                    gCO2eq += wkf["measurements"]["gCO2eq"].sum()
        return gCO2eq

    @property
    def releases(self) -> pd.DataFrame:
        """List of published releases

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        js = self.fs.open_json(
            f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"
        )
        results = []
        for rel in js:
            results.append(
                {
                    "tag": rel["tag_name"],
                    "published": not bool(rel["draft"]),
                    "date": rel["published_at"],
                }
            )
        df = pd.DataFrame(results)
        df = df.sort_values("date").reset_index(drop=1)
        return df

    @property
    def lastreleasedate(self) -> pd.Timestamp:
        """Last release publication date

        Returns
        -------
        :class:`pandas.Timestamp`
        """
        df = self.releases
        return pd.to_datetime(df["date"].max())

    def get_PRtitle(self, pr_num: int) -> str:
        """Get the title of a given PR number

        Parameters
        ----------
        pr_num : int

        Returns
        -------
        str
        """
        js = self.fs.open_json(
            f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls/{pr_num}"
        )
        return js["title"]

    def get_PRmerged_since(self, start_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """List all PR merged since a given date

        Parameters
        ----------
        start_date : pd.Timestamp

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        payload = {
            "state": "closed",
            "sort": "created",
            "per_page": 100,
            "page": 1,
            "direction": "desc",
        }
        # Note that we don't look for other page results and expect less than 100 PRs merged since start_date !
        uri = (
            f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls?"
            + urllib.parse.urlencode(payload)
        )
        js = self.fs.open_json(uri)
        results = []
        for j in js:
            PRmerged = pd.to_datetime(j["merged_at"])
            if PRmerged and PRmerged > pd.to_datetime(start_date, utc=1):
                results.append(
                    {
                        "ID": j["number"],
                        "title": j["title"],
                        "merged": pd.to_datetime(j["merged_at"]),
                    }
                )
        df = pd.DataFrame(results).sort_values("merged").reset_index(drop=1)
        return df

    @property
    def lastPRs(self) -> pd.DataFrame:
        """List all PRs merged since the last release

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`ArgopyCarbon.get_PRmerged_since`, :class:`ArgopyCarbon.lastreleasedate`
        """
        return self.get_PRmerged_since(self.lastreleasedate)

    def footprint_since_last_release(self, with_master: bool = True, errors='ignore') -> float:
        """Compute total carbon footprint since the last release

        Parameters
        ----------
        with_master : bool, default=True
            Should we consider also the 'master' footprint or not.
        errors: Literal, default: ``ignore``
            Define how to handle errors raised during data fetching:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

        Returns
        -------
        float
            Total carbon footprint in gCO2eq, considering all workflows

        See Also
        --------
        :class:`ArgopyCarbon.total_measurements`, :class:`ArgopyCarbon.lastreleasedate`, :class:`ArgopyCarbon.lastPRs`
        """
        df = self.lastPRs
        branches = ["%i/merge" % pr for pr in df["ID"]]
        if with_master:
            branches.append("master")
        return self.total_measurements(branches, start_date=self.lastreleasedate, errors=errors)

    def __repr__(self):
        summary = ["<ArgopyCarbon>"]
        summary.append("Last release date: %s" % self.lastreleasedate)
        summary.append("%i PRs merged since the last release" % len(self.lastPRs))
        summary.append(
            "Workflows analysed: %s"
            % ", ".join(
                ["%s [%s]" % (wkf["Name"], wkf["ID"]) for wkf in self.workflows]
            )
        )
        summary.append(
            "gCO2eq since last release (including 'master' branch'): %0.2f gCO2eq"
            % self.footprint_since_last_release(errors='ignore')
        )
        return "\n- ".join(summary)

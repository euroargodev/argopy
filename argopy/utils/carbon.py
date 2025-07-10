import json
import urllib
import pandas as pd
import copy
from typing import Union, List, Dict, Literal
import logging
from pathlib import Path
import numpy as np

from ..errors import DataNotFound
from . import Registry

log = logging.getLogger("argopy.utils.monitors")


class GreenCoding:
    """GreenCoding API helper class for argopy carbon footprint

    This class uses the `Green-Coding Solutions API <https://api.green-coding.io>`_ to retrieve CI tests energy consumption data.

    This class also uses :class:`Github` to get PRs and release information for argopy.

    Examples
    --------
    .. code-block:: python
        :caption: GreenCoding API

        from argopy.utils import GreenCoding

        GreenCoding().workflows
        GreenCoding().measurements(branch='master', start_date='2025-01-01')
        GreenCoding().measurements(branch='385/merge', start_date='2025-01-01')
        GreenCoding().total_measurements(branches=['master', '385/merge'])

        GreenCoding().footprint_for_release('v1.0.0')
        GreenCoding().footprint_since_last_release()

    .. code-block:: python
        :caption: Get metrics for another repo

        ac = GreenCoding()
        ac.repo = 'argopy-status'
        ac.workflows = [{'ID': '2724029', 'Name': 'API status'}]
        ac.total_measurements()

    """

    owner = "euroargodev"
    """Github owner parameter, default to 'euroargodev'"""

    repo = "argopy"
    """Github repo parameter, default to 'argopy'"""

    workflows = [
        {"ID": "22344160", "Name": "CI tests"},
        {"ID": "25052179", "Name": "CI tests upstream"},
    ]
    """List of github actions workflow parameters"""

    def __init__(self):
        from ..stores import httpstore

        self.fs = httpstore(cache=True)
        self.gh = Github()
        self.URI = Registry(name="GreenCoding API calls")

    def measurements(
        self,
        branch: str = "master",
        start_date: Union[str, pd.Timestamp] = "2024-06-01",
        end_date: Union[str, pd.Timestamp] = None,
        errors: Literal["raise", "ignore", "silent"] = "ignore",
    ) -> List[Dict]:
        """Return measurements for a given branch and date period

        Parameters
        ----------
        branch : str, default='master'
            Name of the branch to retrieve measurements for
        start_date : str, :class:`pandas.Timestamp`, default="2024-06-01"
            Measurements starting date, default to the beginning of measurements.
        end_date : str, :class:`pandas.Timestamp`, default=None
            Measurements ending date, default to today
        errors: Literal, default: ``ignore``
            Define how to handle errors raised during data fetching:
                - ``ignore`` (default): Do not stop processing, simply issue a debug message in logging console
                - ``raise``: Raise any error encountered
                - ``silent``:  Do not stop processing and do not issue log message

        Returns
        -------
        List[Dict] :
            List of workflows name, ID and measurements as :class:`pandas.DataFrame`

        See Also
        --------
        :class:`GreenCoding.workflows`
        """
        if end_date is None:
            end_date = pd.to_datetime("now", utc=True)
        else:
            end_date = pd.to_datetime(end_date)
        if start_date is None:
            start_date = pd.to_datetime("2020-03-17", utc=True)  # 1st argopy release
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
            "Latitude",
            "Longitude",
            "City",
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
            "Latitude",
            "Longitude",
            "City",
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
            self.URI.commit(uri)

            try:
                data = self.fs.open_json(uri)

                # Remove out of range values:
                for ii, item in enumerate(data["data"]):
                    if item[names.index("Energy")] / 1e6 > 20000:
                        data["data"].pop(ii)
                    elif item[names.index("gCO2eq")] / 1e6 > 1000:
                        data["data"].pop(ii)

                df = pd.DataFrame(data["data"], columns=names)
                df["Duration"] = df["Duration"] / 1e6  # seconds
                df["gCO2eq"] = df["gCO2eq"] / 1e6  # gCO2eq
                df["Energy"] = df["Energy"] / 1e6  # Joules

                df2 = df[["Workflow_Step", "Energy", "Duration", "gCO2eq"]]
                df2 = df2.groupby("Workflow_Step").sum()

                results[irow]["measurements"] = df2

            except json.JSONDecodeError:
                msg = (
                    "No data returned, probably because the branch '%s' was not found at %s"
                    % (branch, uri)
                )
                if errors == "raise":
                    from ..errors import DataNotFound

                    raise DataNotFound(msg)
                elif errors == "ignore":
                    log.debug(msg)

            except Exception:
                if errors == "raise":
                    raise
                elif errors == "ignore":
                    log.error("Error: {e}")

        return results

    def total_measurements(self, branches: List[str] = ["master"], **kwargs) -> float:
        """Compute the cumulated measurements of gCO2eq for a list of branches and workflows

        Parameters
        ----------
        branches : List[str], default = ['master']
            List of branches to retrieve measurements for.
            Note that for a given merged PR number, the branch name is ``<PR>/merged``.
        **kwargs:
            Other Parameters are passed to :class:`GreenCoding.measurements`

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

    def footprint_for_release(
        self, release: str = None, with_master: bool = True
    ) -> pd.DataFrame:
        """Compute total carbon footprint for a given release

        Parameters
        ----------
        release : str, optional
            Tag of the release. E.g. 'v1.0.0'. By default, we use the last release
        with_master : bool, default=True
            Should we consider also the 'master' branch footprint or not.

        Returns
        -------
        float
            Total carbon footprint in gCO2eq, considering all workflows

        """
        release = self.gh.lastrelease_tag if release is None else release
        df = self.gh.ls_PRmerged_in_release(release=release)
        value = 0
        for ipr, pr in df.iterrows():
            branches = ["%i/merge" % pr["ID"]]
            meas = self.total_measurements(
                branches, start_date=pr["created"], end_date=pr["merged"]
            )
            value += meas
        if with_master:
            value += self.total_measurements(
                ["master"], start_date=df["created"].min(), end_date=df["merged"].max()
            )
        return value

    def footprint_since_last_release(
        self, with_master: bool = True, errors="ignore"
    ) -> float:
        """Compute total carbon footprint since the last release

        Parameters
        ----------
        with_master : bool, default=True
            Should we consider also the 'master' branch footprint or not.
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
        :class:`GreenCoding.total_measurements`, :class:`Github.lastrelease_date`, :class:`Github.ls_PRmerged_since_last_release`
        """
        df = self.gh.ls_PRmerged_since_last_release
        branches = ["%i/merge" % pr for pr in df["ID"]]
        if with_master:
            branches.append("master")
        return self.total_measurements(
            branches, start_date=self.gh.lastrelease_date, errors=errors
        )

    def footprint_all_releases(self):
        """Footprint for all releases"""
        releases = self.gh.releases
        footprint = []
        for ii, release in releases.iterrows():
            try:
                footprint.append(self.footprint_for_release(release["tag"]))
            except Exception:
                footprint.append(0.0)
                pass
        releases["gCO2eq"] = footprint
        # releases['gCO2eq'].fillna(0.0, inplace=True)
        return releases

    def footprint_baseline(
        self,
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None,
        errors="ignore",
    ):
        """Compute the 'baseline' footprint

        The 'baseline' footprint is from PRs that are not merged. PRs status can be 'open' or 'closed'.
        All details are available on the `documentation page <https://argopy.readthedocs.io/en/latest/energy.html>`_.

        Parameters
        ----------
        start_date : str, :class:`pandas.Timestamp`, default=None
            Measurements starting date, default to start of measurements
        end_date : str, :class:`pandas.Timestamp`, default=None
            Measurements ending date, default to today
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
        :class:`GreenCoding.for_release`, :class:`GreenCoding.total_measurements`
        """
        df = self.gh.ls_PRbaseline(start_date=start_date, end_date=end_date)
        branches = ["%i/merge" % pr for pr in df["ID"]]
        return self.total_measurements(
            branches, start_date=start_date, end_date=end_date, errors=errors
        )

    def __repr__(self):
        summary = [f"<GreenCoding.{self.owner}.{self.repo}>"]
        summary.append("Last release date: %s" % self.gh.lastrelease_date)
        summary.append(
            "%i PRs merged since the last release"
            % len(self.gh.ls_PRmerged_since_last_release)
        )
        summary.append(
            "Workflows analysed: %s"
            % ", ".join(
                ["%s [%s]" % (wkf["Name"], wkf["ID"]) for wkf in self.workflows]
            )
        )
        summary.append(
            "gCO2eq since last release (including 'master' branch'): %0.2f"
            % self.footprint_since_last_release(errors="ignore")
        )
        return "\n- ".join(summary)

    @staticmethod
    def shieldsio_badge(
        value: float, label: str = "Total carbon emitted [gCO2eq]"
    ) -> str:
        """Insert value in a Shields.io badge and return url

        Insert total measurement value into a shields.io badge url

        Parameters
        ----------
        value : float
            The carbon footprint value
        label: str, default: 'Total carbon emitted [gCO2eq]'
            The badge label to use

        Returns
        -------
        str
        """
        payload = {
            "style": "plastic",
            "labelColor": "grey",
        }
        t = lambda t: urllib.parse.quote(t)  # noqa: E731
        uri = "https://img.shields.io/badge/%s-%s-%s?" % (
            t(label),
            t("%0.2f" % value),
            "black",
        ) + urllib.parse.urlencode(payload)
        return uri

    @staticmethod
    def shieldsio_endpoint(
        value: float, outfile: Path = None, label: str = "Total carbon emitted [gCO2eq]"
    ) -> Path:
        """Insert value in a Shields.io json endpoint file

        This is used by the argopy monitoring repository.

        See live results at:

        https://github.com/euroargodev/argopy-status?tab=readme-ov-file#energy-impact

        Parameters
        ----------
        value : float
            The carbon footprint value
        outfile: :class:`pathlib.Path`, default: None
            Path toward json file
        label: str, default: 'Total carbon emitted [gCO2eq]'
            The badge label to use

        Returns
        -------
        :class:`pathlib.Path`
        """

        def save_to_json(label, message, outfile):
            """Save a shields.io badge endpoint to a json file"""
            data = {}
            data["schemaVersion"] = 1
            data["label"] = label
            data["labelColor"] = "grey"
            data["message"] = message
            data["color"] = "black"
            with open(outfile, "w") as f:
                json.dump(data, f)
            return outfile

        return save_to_json(
            label=label,
            message="%0.2f" % value,
            outfile=Path(outfile),
        )


class Github:
    """Github repository helper class

    This class uses the `Github API <https://docs.github.com/en/rest>`_, to get PRs and release information.

    It is primarily meant to be used by :class:`GreenCoding`.

    Examples
    --------
    .. code-block:: python
        :caption: Github API

        from argopy.utils import Github

        Github().releases
        Github().lastrelease_date
        Github().lastrelease_tag
        Github().get_PRtitle(385)
        Github().ls_PRs()
        Github().ls_PRs('2025-01-01')
        Github().ls_PRmerged('2025-01-01')
        Github().ls_PRmerged('2024-01-01', '2025-01-01')
        Github().ls_PRmerged_since_last_release
        Github().ls_PRmerged_in_release('v1.0.0')
    """

    owner = "euroargodev"
    """Github owner parameter, default to 'euroargodev'"""

    repo = "argopy"
    """Github repo parameter, default to 'argopy'"""

    _js_data_pulls = None
    """Place holder for json data returned from the Github API pulls endpoint"""

    def __init__(self, repo: str = "argopy", owner: str = "euroargodev"):
        from ..stores import httpstore

        self.fs = httpstore(cache=True)
        self.owner = owner
        self.repo = repo

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
                    "tag": rel["tag_name"].strip(),
                    "published": not bool(rel["draft"]),
                    "date": rel["published_at"],
                }
            )
        df = pd.DataFrame(results)
        df = df.sort_values("date").reset_index(drop=1)
        return df

    @property
    def lastrelease_date(self) -> pd.Timestamp:
        """Last release publication date

        Returns
        -------
        :class:`pandas.Timestamp`
        """
        df = self.releases
        return pd.to_datetime(df["date"].max())

    @property
    def lastrelease_tag(self) -> str:
        """Last release tag name

        Returns
        -------
        str
        """
        df = self.releases
        return df["tag"].iloc[-1]

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

    @property
    def _json_pulls(self):
        """Return json data from Github API pulls endpoint

        All results are returned, i.e. from all pages

        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#list-pull-requests
        """

        def list_uri(max_page=5):
            uri = []
            for page in range(1, max_page):
                payload = {
                    "state": "all",  # open/closed
                    "per_page": 100,
                    "page": page,
                    "sort": "update",
                    "direction": "asc",
                }
                url = (
                    f"https://api.github.com/repos/{self.owner}/{self.repo}/pulls?"
                    + urllib.parse.urlencode(payload)
                )
                uri.append(url)
            return uri

        # [print(u) for u in list_uri()]
        if self._js_data_pulls is None:
            pages = self.fs.open_mfjson(list_uri())
            self._js_data_pulls = []
            for page in pages:
                for rec in page:
                    self._js_data_pulls.append(rec)

        return self._js_data_pulls

    def ls_PRs(
        self,
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None,
        date: Literal["created", "updated", "merged"] = "created",
    ) -> pd.DataFrame:
        """List all PRs, possibly between 2 dates

        Parameters
        ----------
        start_date : :class:`pandas.Timestamp`, default = None
        end_date : :class:`pandas.Timestamp`, default = None
        date: Literal['created', 'updated', 'merged'], default = 'created'
            Date filter key, use to apply bounds set by start_date and/or end_date

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        start_date = (
            pd.to_datetime(start_date, utc=True)
            if start_date is not None
            else pd.to_datetime(start_date, utc=True)
        )
        end_date = (
            pd.to_datetime(end_date, utc=True)
            if end_date is not None
            else pd.to_datetime(end_date, utc=True)
        )

        results = []
        for j in self._json_pulls:
            merged = pd.to_datetime(j["merged_at"])
            ismerged = merged is not None
            created = pd.to_datetime(j["created_at"])
            updated = pd.to_datetime(j["updated_at"])

            sorting_key = "ID"
            date_filter = False
            if start_date is not None and end_date is not None:
                if date == "created":
                    sorting_key = "created"
                    if created >= start_date and created <= end_date:
                        date_filter = True
                elif date == "updated":
                    sorting_key = "updated"
                    if updated >= start_date and updated <= end_date:
                        date_filter = True
                elif date == "merged" and ismerged:
                    sorting_key = "merged"
                    if merged >= start_date and merged <= end_date:
                        date_filter = True
            elif start_date is not None:
                if date == "created":
                    sorting_key = "created"
                    if created >= start_date:
                        date_filter = True
                elif date == "updated":
                    sorting_key = "updated"
                    if updated >= start_date:
                        date_filter = True
                elif date == "merged" and ismerged:
                    sorting_key = "merged"
                    if merged >= start_date:
                        date_filter = True
            elif end_date is not None:
                if date == "created":
                    sorting_key = "created"
                    if created >= end_date:
                        date_filter = True
                elif date == "updated":
                    sorting_key = "updated"
                    if updated >= end_date:
                        date_filter = True
                elif date == "merged" and ismerged:
                    sorting_key = "merged"
                    if merged >= end_date:
                        date_filter = True
            else:
                date_filter = True

            if date_filter:
                results.append(
                    {
                        "ID": j["number"],
                        "title": j["title"],
                        "created": created,
                        "updated": updated,
                        "merged": merged,
                        "state": j["state"],
                    }
                )
        if len(results) == 0:
            raise DataNotFound()
        else:
            df = pd.DataFrame(results).sort_values(sorting_key).reset_index(drop=1)
            return df

    def ls_PRmerged(
        self,
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """List merged PRs

        Parameters
        ----------
        start_date : :class:`pandas.Timestamp`
        end_date : :class:`pandas.Timestamp`

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`Github.ls_PRs`
        """
        df = self.ls_PRs(start_date=start_date, end_date=end_date, date="merged")
        df = df[np.logical_and(df["merged"].notnull(), df["state"] == "closed")]
        df = df.sort_values("merged").reset_index(drop=1)
        return df

    @property
    def ls_PRmerged_since_last_release(self) -> pd.DataFrame:
        """List PRs merged since the last release

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`Github.ls_PRmerged`, :class:`Github.ls_PRmerged_in_release`, :class:`GreenCoding.lastrelease_date`
        """
        return self.ls_PRmerged(start_date=self.lastrelease_date)

    def ls_PRmerged_in_release(self, release: str) -> pd.DataFrame:
        """List PRs included in a given release

        Parameters
        ----------
        release : str
            The release tag name to use

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`Github.ls_PRmerged`, :class:`Github.ls_PRmerged_since_last_release`
        """
        df_rel = self.releases

        if release not in df_rel["tag"].values:
            raise ValueError("Release '%s' does not exist !" % release)

        # Get this release publication date:
        end_date = df_rel[df_rel["tag"] == release]["date"].values[0]

        # Get the previous release publication date:
        start_date = df_rel.iloc[df_rel[df_rel["tag"] == release].index[0] - 1]["date"]

        if release == df_rel["tag"].iloc[0]:
            raise ValueError("No PRs for the initial release !")

        return self.ls_PRmerged(start_date=start_date, end_date=end_date)

    def ls_PRbaseline(
        self,
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """List all PRs in the baseline

        A *baseline* PR is a PR is not merged back to the master branch, hence such a PR is not included in any release.

        Parameters
        ----------
        start_date : :class:`pandas.Timestamp`, default = None
        end_date : :class:`pandas.Timestamp`, default = None

        Returns
        -------
        :class:`pandas.DataFrame`

        See Also
        --------
        :class:`Github.ls_PRs`
        """
        df = self.ls_PRs(start_date=start_date, end_date=end_date, date="updated")
        df = df[~df["merged"].notnull()]
        return df

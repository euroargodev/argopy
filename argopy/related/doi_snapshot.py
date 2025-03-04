import pandas as pd
import numpy as np
import warnings
from typing import Union

# from matplotlib.colors import to_hex
# from IPython.display import IFrame

from ..stores import httpstore


class DOIrecord:
    """Meta-data holder for an Argo GDAC snapshot DOI record

    This is a low-level class that is not intended to be instantiated directly.

    Please use the :class:`ArgoDOI` instead.

    Examples
    --------
    .. code-block:: python
        :caption: API description

        d = DOIrecord()
        d = DOIrecord('42182')
        d = DOIrecord('42182#103075')
        d = DOIrecord(hashtag='103075')
        d = DOIrecord(hashtag='103088')

        d.doi
        d.dx
        d.isvalid
        d.date
        d.network
        d.data
        d.file

    """
    root = ""

    def __init__(
        self,
        doi: str = "10.17882/42182",
        hashtag: str = None,
        fs: httpstore = None,
        autoload: bool = True,
        api_root: str = "https://www.seanoe.org/api/",
    ):
        self.api_root = api_root
        self._fs = fs  # A httpstore will be created if necessary if self.load() is called
        self._data = None

        self._doi = doi
        self._hashtag = hashtag
        if "#" in doi:
            self._doi = doi.split("#")[0]
            self._hashtag = doi.split("#")[-1]

        if autoload:
            self.load()

    @property
    def doi(self) -> str:
        """DOI component (without hashtag)"""
        return self._doi

    @property
    def hashtag(self) -> str:
        """Hashtag of the full doi"""
        return self._hashtag

    @property
    def dx(self) -> str:
        """DOI url"""
        return "https://dx.doi.org/%s" % str(self)

    def isvalid(self) -> bool:
        return "42182" in self.doi

    @property
    def data(self) -> dict:
        """DOI record meta-data holder

        Trigger data (down)load if not available
        """
        if self._data is None:
            self.load()
        return self._data

    @property
    def date(self) -> pd.Timestamp:
        """Date associated with the DOI record"""
        return self.data["date"]

    @property
    def network(self) -> str:
        """Network of the Argo data pointed by the DOI

        Returns
        -------
        str: 'core+BGC+deep' or 'BGC'
        """
        return "BGC" if "BGC" in self.data["title"] else "core+BGC+deep"

    @property
    def file(self) -> list:
        """Return a pretty list of files properties associated with this DOI"""
        results = []
        for f in self.data["files"]:
            r = {"openAccess": bool(f["openAccess"])}
            if bool(f["openAccess"]):
                r["path"] = f["fileUrl"]
            else:
                r["path"] = None
            r["update"] = pd.to_datetime(f["lastUpdateDate"])
            r["date"] = pd.to_datetime(f["fragment"]["date"])
            r["size"] = f["size"]
            r["network"] = "BGC" if "BGC" in f["fragment"]["title"] else "core+BGC+deep"
            results.append(r)
        return results

    @property
    def uri(self) -> str:
        """url to API call to retrieve DOI data"""
        if self.hashtag is None:
            url = "find-by-id/{id}".format
        else:
            url = "find-by-fragment/{id}?fragmentId={hashtag}".format
        return self.api_root + url(id=self.doi.split("/")[-1], hashtag=self.hashtag)

    def __str__(self):
        # txt = "%s/%s" % (self.root, self.doi)
        txt = "%s" % (self.doi)
        if self.hashtag is not None:
            txt = "%s#%s" % (txt, self._hashtag)
        return txt

    def _process_data(self, data: dict) -> dict:
        """Synthetic dict from data return by API"""
        Nfiles = len(data["files"])
        if Nfiles > 1:
            # Sort files resources by date (most recent first)
            data["files"].sort(
                key=lambda x: x.get("fragment").get("date"), reverse=True
            )

        return {
            "title": data["title"]["en"],
            "date": pd.to_datetime(data["date"]),
            "authors": data["authors"],
            "files": data["files"],
            "Nfiles": Nfiles,
            # 'description': data['description'],
            # 'keywords': data['keywords'],
            # 'licenceUrl': data['licenceUrl'],
        }

    def load(self, cache: bool = False):
        """Load DOI record data from API call"""
        if self._data is None:
            if self._fs is None:
                self._fs = httpstore(cache=cache)

            data = self._fs.open_json(self.uri)
            self._data = self._process_data(data)

        return self

    def from_dict(self, d: dict):
        """Load DOI record data from a dictionary"""
        if (
            "title" in d
            and "en" in d["title"]
            and "date" in d
            and "authors" in d
            and "files" in d
        ):
            self._data = self._process_data(d)
        return self

    def search(self, **kwargs):
        raise ValueError("")

    def _repr_file(self, file, with_label=False) -> str:
        """Return a pretty string from a single file dict"""
        def sizeof_fmt(num, suffix="B"):
            for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
                if abs(num) < 1024.0:
                    return f"{num:3.1f}{unit}{suffix}"
                num /= 1024.0
            return f"{num:.1f}Yi{suffix}"

        summary = []
        if with_label:
            summary.append("%s" % file["label"]["en"])

        if bool(file["openAccess"]):
            summary.append("%s" % file["fileUrl"])
        else:
            summary.append("%s" % file["fileName"])

        attrs = []
        attrs.append("%s" % sizeof_fmt(file["size"]))
        attrs.append("openAccess=%s" % file["openAccess"])
        summary.append("(%s)" % (", ".join(attrs)))

        return " ".join(summary)

    def __repr__(self):
        summary = ["<argopy.DOI.record>"]
        summary.append("DOI: %s" % self.__str__())
        if self._data is not None:
            summary.append("Title: %s" % self.data["title"])
            summary.append("Date: %s" % self.date.strftime("%Y-%m-%d"))
            summary.append("Network: %s" % self.network)
            summary.append("Link: %s" % self.dx)

            if self.data["Nfiles"] == 1:
                summary.append("File: %s" % self._repr_file(self.data["files"][0]))
            else:
                summary.append("File: %i files in total" % (self.data["Nfiles"]))

                summary.append("Files for core+BGC+deep:")
                ifound = 0
                for ii, f in enumerate(self.data["files"]):
                    if "BGC" not in f["fragment"]["title"] and ifound < 10:
                        summary.append(
                            "     - #%s %s"
                            % (f["id"], self._repr_file(f, with_label=True))
                        )
                        ifound += 1

                summary.append("Files for BGC only:")
                ifound = 0
                for ii, f in enumerate(self.data["files"]):
                    if "BGC" in f["fragment"]["title"] and ifound < 10:
                        summary.append(
                            "     - #%s %s"
                            % (f["id"], self._repr_file(f, with_label=True))
                        )
                        ifound += 1

        return "\n".join(summary)

    # @property
    # def html(self) -> str:
    #     fs = 12
    #
    #     def td_msg(bgcolor, txtcolor, txt):
    #         style = "background-color:%s;" % to_hex(bgcolor, keep_alpha=True)
    #         style += "border-width:0px;"
    #         style += "padding: 2px 2px 2px 0px;"
    #         style += "text-align:left;"
    #         style += "color:%s" % to_hex(txtcolor, keep_alpha=True)
    #         return "<td style='%s'>%s</td>" % (style, str(txt))
    #
    #     def td_a(bgcolor, txtcolor, txt, link):
    #         style = "background-color:%s;" % to_hex(bgcolor, keep_alpha=True)
    #         style += "border-width:0px;"
    #         style += "padding: 2px 0px 2px 5px;"
    #         style += "text-align:right;"
    #         style += "color:%s" % to_hex(txtcolor, keep_alpha=True)
    #         return "<td style='%s'><a href='%s'>%s</a></td>" % (style, link, str(txt))
    #
    #     td_empty = "<td style='border-width:0px;padding: 2px 5px 2px 5px;text-align:left'>&nbsp;</td>"
    #
    #     html = []
    #     html.append(
    #         "<table style='border-collapse:collapse;border-spacing:0;font-size:%ipx'>"
    #         % fs
    #     )
    #     html.append("<tbody>")
    #
    #     rows = []
    #
    #     # 1st row:
    #     cols = []
    #     cols.append(td_msg("dimgray", "w", "doi: "))
    #     cols.append(td_msg("green", "w", "%s/" % self.root))
    #     cols.append(td_msg("yellowgreen", "w", self.doi))
    #     if self.hashtag is not None:
    #         cols.append(td_msg("darkorange", "w", "#%s" % self.hashtag))
    #     cols.append(td_a("white", "w", "↗", self.dx))
    #     cols.append(td_empty)
    #     rows.append("<tr>%s</tr>" % "\n".join(cols))
    #
    #     #         # 2nd row (if data have been loaded):
    #     #         if self._data is not None:
    #     #             cols = []
    #     #             cols.append(td_msg('dimgray', 'w', "Title: "))
    #     #             cols.append(td_msg('white', 'w', "%s" % self.data['title']))
    #     #             # cols.append(td_msg('yellowgreen', 'w', self.doi))
    #     #             # if self.hashtag is not None:
    #     #             #     cols.append(td_msg("darkorange", 'w', "#%s" % self.hashtag))
    #     #             # cols.append(td_a("white", 'w', "↗", self.dx))
    #     #             # cols.append(td_empty)
    #     #             rows.append("<tr>%s</tr>" % "\n".join(cols))
    #
    #     #         print(rows)
    #     #         # Fix colspan:
    #     #         Nrows = np.max([len(r.split("<td ")) for r in rows])
    #     #         print(Nrows)
    #     #         rowss = []
    #     #         for r in rows:
    #     #             rowss.append(r.replace("<tr>", "<tr colspan='%i'>" % Nrows))
    #     #         print(rowss)
    #
    #     # Finalize
    #     html.append("\n".join(rows))
    #     html.append("</tbody>")
    #     html.append("</table>")
    #     html = "\n".join(html)
    #     return html

    # def _repr_html_(self):
    #     return self.html


class ArgoDOI:
    """Argo GDAC snapshot DOI access and discovery

    Examples
    --------
    .. code-block:: python
        :caption: Load DOI meta-data

        from argopy import ArgoDOI

        doi = ArgoDOI()  # If you don't know where to start, just load the primary Argo DOI record
        doi = ArgoDOI('95141')  # To point directly to a snapshot ID
        doi = ArgoDOI(hashtag='95141')
        doi = ArgoDOI(fs=httpstore(cache=True))

    .. code-block:: python
        :caption: Searching for a specific DOI snapshot

        # Return doi closest to a given date:
        ArgoDOI().search('2020-02')

        # Return doi closest to a given date for a specific network:
        ArgoDOI().search('2020-02', network='BGC')

    .. code-block:: python
        :caption: Working with DOIs

        doi = ArgoDOI('95141')

        doi.download()  # Trigger download of the DOI file
        doi.file  # Easy to read list of file(s) associated with a DOI record
        doi.dx  # http link toward the DOI snapshot webpage

    """

    def __init__(self,
                 hashtag=None,
                 fs=None,
                 cache=True):
        self._fs = fs if isinstance(fs, httpstore) else httpstore(cache=cache)
        if hashtag is not None and '42182#' in hashtag:
            hashtag = hashtag.split('42182#')[-1]
        self._doi = DOIrecord(hashtag=hashtag, fs=self._fs, autoload=True)

    @property
    def doi(self) -> str:
        """DOI component (without hashtag)"""
        return str(self._doi)

    def __repr__(self):
        summary = self._doi.__repr__().split("\n")
        summary[0] = '<argopy.DOI>'
        return "\n".join(summary)

    def dates(self, network: str = None) -> dict:
        """Mapping of DOI snapshot hashtag(s) to their publication date(s)

        Parameters
        ----------
        network: str, optional
            Allows to specify a network, like 'BGC'.

        Returns
        -------
        dict
            Dictionary where keys are DOI hashtag and values are publication dates as :class:`pandas.Timestamp`
        """
        d = {}
        network = self._doi.network if network is None else network
        if network == "BGC":
            for f in self._doi.data["files"]:
                if "BGC" in f["fragment"]["title"]:
                    d.update({int(f["id"]): pd.to_datetime(f["fragment"]["date"])})
        else:
            for f in self._doi.data["files"]:
                if "BGC" not in f["fragment"]["title"]:
                    d.update({int(f["id"]): pd.to_datetime(f["fragment"]["date"])})
        return d

    def search(self, date: Union[str, pd.Timestamp], network: str = None) -> DOIrecord:
        """Search the DOI record the closest to a given date

        Parameters
        ----------
        date: str, :class:`pandas.Timestamp`
            Date to search a DOI for
        network: str, optional
            Allows to specify a network, like 'BGC'

        Returns
        -------
        :class:`argopy.related.doi_snapshot.DOIrecord`
        """
        dates = self.dates(network=network)
        target = pd.to_datetime(date, utc=True)
        close = list(dates.values())[
            np.argmin(np.abs([target - dates[d] for d in dates]))
        ]
        found = [d for d in dates if dates[d] == close]
        results = []
        if len(found) > 0:
            for f in found:
                results.append(DOIrecord(hashtag=f, fs=self._fs))
        if len(results) == 1:
            if (close - target).days > 30:
                warnings.warn(
                    "This snapshot is more than 30 days off your search dates !"
                )
            return results[0]
        else:
            return results

    @property
    def file(self) -> list:
        """DOI tar.gz file properties"""
        return self._doi.file

    @property
    def dx(self) -> str:
        """DOI url"""
        return self._doi.dx

    def download(self):
        """Trigger download of a DOI tar.gz file

        This will simply make the web browser to open the DOI file.
        """
        flist = self.file
        if len(flist) > 1:
            warnings.warn("For safety reasons, we don't trigger download of a DOI when it has more than one file. This is probably happening because you did not specified a hashtag to your ArgoDOI instance.")
        else:
            import webbrowser
            webbrowser.open_new(self.file[0]['path'])

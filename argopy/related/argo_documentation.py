import pandas as pd
from functools import lru_cache
from ..stores import httpstore
from ..options import OPTIONS


class ArgoDocs:
    """ADMT documentation helper class

    Examples
    --------
    >>> ArgoDocs().list
    >>> ArgoDocs().search("CDOM")
    >>> ArgoDocs().search("CDOM", where='abstract')

    >>> ArgoDocs(35385)
    >>> ArgoDocs(35385).ris
    >>> ArgoDocs(35385).abstract
    >>> ArgoDocs(35385).show()
    >>> ArgoDocs(35385).open_pdf()
    >>> ArgoDocs(35385).open_pdf(page=12)

    """
    _catalogue = [
        {
            "category": "Argo data formats",
            "title": "Argo user's manual",
            "doi": "10.13155/29825",
            "id": 29825
        },
        {
            "category": "Quality control",
            "title": "Argo Quality Control Manual for CTD and Trajectory Data",
            "doi": "10.13155/33951",
            "id": 33951
        },
        {
            "category": "Quality control",
            "title": "Argo quality control manual for dissolved oxygen concentration",
            "doi": "10.13155/46542",
            "id": 46542
        },
        {
            "category": "Quality control",
            "title": "Argo quality control manual for biogeochemical data",
            "doi": "10.13155/40879",
            "id": 40879
        },
        {
            "category": "Quality control",
            "title": "BGC-Argo quality control manual for the Chlorophyll-A concentration",
            "doi": "10.13155/35385",
            "id": 35385
        },
        {
            "category": "Quality control",
            "title": "BGC-Argo quality control manual for nitrate concentration",
            "doi": "10.13155/84370",
            "id": 84370
        },
        {
            "category": "Quality control",
            "title": "Quality control for BGC-Argo radiometry",
            "doi": "10.13155/62466",
            "id": 62466
        },
        {
            "category": "Cookbooks",
            "title": "Argo DAC profile cookbook",
            "doi": "10.13155/41151",
            "id": 41151
        },
        {
            "category": "Cookbooks",
            "title": "Argo DAC trajectory cookbook",
            "doi": "10.13155/29824",
            "id": 29824
        },
        {
            "category": "Cookbooks",
            "title": "DMQC Cookbook for Core Argo parameters",
            "doi": "10.13155/78994",
            "id": 78994
        },
        {
            "category": "Cookbooks",
            "title": "Processing Argo oxygen data at the DAC level",
            "doi": "10.13155/39795",
            "id": 39795
        },
        {
            "category": "Cookbooks",
            "title": "Processing Bio-Argo particle backscattering at the DAC level",
            "doi": "10.13155/39459",
            "id": 39459
        },
        {
            "category": "Cookbooks",
            "title": "Processing BGC-Argo chlorophyll-A concentration at the DAC level",
            "doi": "10.13155/39468",
            "id": 39468
        },
        {
            "category": "Cookbooks",
            "title": "Processing Argo measurement timing information at the DAC level",
            "doi": "10.13155/47998",
            "id": 47998
        },
        {
            "category": "Cookbooks",
            "title": "Processing BGC-Argo CDOM concentration at the DAC level",
            "doi": "10.13155/54541",
            "id": 54541
        },
        {
            "category": "Cookbooks",
            "title": "Processing Bio-Argo nitrate concentration at the DAC Level",
            "doi": "10.13155/46121",
            "id": 46121
        },
        {
            "category": "Cookbooks",
            "title": "Processing BGC-Argo Radiometric data at the DAC level",
            "doi": "10.13155/51541",
            "id": 51541
        },
        {
            "category": "Cookbooks",
            "title": "Processing BGC-Argo pH data at the DAC level",
            "doi": "10.13155/57195",
            "id": 57195
        },
        {
            "category": "Cookbooks",
            "title": "Description of the Argo GDAC File Checks: Data Format and Consistency Checks",
            "doi": "10.13155/46120",
            "id": 46120
        },
        {
            "category": "Cookbooks",
            "title": "Description of the Argo GDAC File Merge Process",
            "doi": "10.13155/52154",
            "id": 52154
        },
        {
            "category": "Cookbooks",
            "title": "BGC-Argo synthetic profile file processing and format on Coriolis GDAC",
            "doi": "10.13155/55637",
            "id": 55637
        },
        {
            "category": "Cookbooks",
            "title": "Argo GDAC cookbook",
            "doi": "10.13155/46202",
            "id": 46202
        }
    ]

    class RIS:
        """RIS file structure from TXT file"""

        def __init__(self, file=None, fs=None):
            self.record = None
            self.fs = fs
            if file:
                self.parse(file)

        def parse(self, file):
            """Parse input file"""
            # log.debug(file)

            with self.fs.open(file, 'r', encoding="utf-8") as f:
                TXTlines = f.readlines()
            lines = []
            # Eliminate blank lines
            for line in TXTlines:
                line = line.strip()
                if len(line) > 0:
                    lines.append(line)
            TXTlines = lines

            #
            record = {}
            for line in TXTlines:
                # print("\n>", line)
                if len(line) > 2:
                    if line[2] == " ":
                        tag = line[0:2]
                        field = line[3:]
                        # print("ok", {tag: field})
                        record[tag] = [field]
                    else:
                        # print("-", line)
                        record[tag].append(line)
                elif len(line) == 2:
                    record[line] = []
                # else:
                # print("*", line)

            for key in record.keys():
                record[key] = "; ".join(record[key])

            self.record = record

    @lru_cache
    def __init__(self, docid=None, cache=False):
        self.docid = None
        self._ris = None
        self._risfile = None
        self._fs = httpstore(cache=cache, cachedir=OPTIONS['cachedir'])
        self._doiserver = "https://dx.doi.org"
        self._archimer = "https://archimer.ifremer.fr"

        if isinstance(docid, int):
            if docid in [doc['id'] for doc in self._catalogue]:
                self.docid = docid
            else:
                raise ValueError("Unknown document id")
        elif isinstance(docid, str):
            start_with = lambda f, x: f[0:len(x)] == x if len(x) <= len(f) else False  # noqa: E731
            if start_with(docid, '10.13155/') and docid in [doc['doi'] for doc in self._catalogue]:
                self.docid = [doc['id'] for doc in self._catalogue if docid == doc['doi']][0]
            else:
                raise ValueError("'docid' must be an integer or a valid Argo DOI")

    def __repr__(self):
        summary = ["<argopy.ArgoDocs>"]
        if self.docid is not None:
            doc = [doc for doc in self._catalogue if doc['id'] == self.docid][0]
            summary.append("Title: %s" % doc['title'])
            summary.append("DOI: %s" % doc['doi'])
            summary.append("url: https://dx.doi.org/%s" % doc['doi'])
            summary.append("last pdf: %s" % self.pdf)
            if 'AF' in self.ris:
                summary.append("Authors: %s" % self.ris['AF'])
            summary.append("Abstract: %s" % self.ris['AB'])
        else:
            summary.append("- %i documents with a DOI are available in the catalogue" % len(self._catalogue))
            summary.append("- Use the method 'search' to find a document id")
            summary.append("- Use the property 'list' to check out the catalogue")
        return "\n".join(summary)

    @property
    def list(self):
        """List of all available documents as a :class:`pandas.DataFrame`"""
        return pd.DataFrame(self._catalogue)

    @property
    def js(self):
        """Internal json record for a document"""
        if self.docid is not None:
            return [doc for doc in self._catalogue if doc['id'] == self.docid][0]
        else:
            raise ValueError("Select a document first !")

    @property
    def ris(self):
        """RIS record of a document"""
        if self.docid is not None:
            if self._ris is None:
                # Fetch RIS metadata for this document:
                import re
                file = self._fs.download_url("%s/%s" % (self._doiserver, self.js['doi']))
                x = re.search(r'<a target="_blank" href="(https?:\/\/([^"]*))"\s+([^>]*)rel="nofollow">TXT<\/a>',
                              str(file))
                export_txt_url = x[1].replace("https://archimer.ifremer.fr", self._archimer)
                self._risfile = export_txt_url
                self._ris = self.RIS(export_txt_url, fs=self._fs).record
            return self._ris
        else:
            raise ValueError("Select a document first !")

    @property
    def abstract(self):
        """Abstract of a document"""
        if self.docid is not None:
            return self.ris['AB']
        else:
            raise ValueError("Select a document first !")

    @property
    def pdf(self):
        """Link to the online pdf version of a document"""
        if self.docid is not None:
            return self.ris['UR']
        else:
            raise ValueError("Select a document first !")

    def show(self, height=800):
        """Insert document in pdf in a notebook cell

        Parameters
        ----------
        height: int
            Height in pixels of the cell
        """
        if self.docid is not None:
            from IPython.core.display import HTML
            return HTML(
                '<embed src="%s" type="application/pdf" width="100%%" height="%ipx" />' % (self.ris['UR'], height))
        else:
            raise ValueError("Select a document first !")

    def open_pdf(self, page=None, url_only=False):
        """Open document in new browser tab

        Parameters
        ----------
        page: int, optional
            Open directly a specific page number
        """
        url = self.pdf
        url += '#view=FitV&pagemode=thumbs'
        if page:
            url += '&page=%i' % page
        if self.docid is not None:
            if not url_only:
                import webbrowser
                webbrowser.open_new(url)
            else:
                return url
        else:
            raise ValueError("Select a document first !")

    def search(self, txt, where='title'):
        """Search for string in all documents title or abstract

        Parameters
        ----------
        txt: str
        where: str, default='title'
            Where to search, can be 'title' or 'abstract'

        Returns
        -------
        list

        """
        results = []
        for doc in self.list.iterrows():
            docid = doc[1]['id']
            if where == 'title':
                if txt.lower() in ArgoDocs(docid).js['title'].lower():
                    results.append(docid)
            elif where == 'abstract':
                if txt.lower() in ArgoDocs(docid).abstract.lower():
                    results.append(docid)
        return results

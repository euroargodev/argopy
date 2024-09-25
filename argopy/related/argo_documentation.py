import os
import json
import pandas as pd
from functools import lru_cache
import requests

from ..stores import httpstore, memorystore
from ..options import OPTIONS
from .utils import path2assets


# Load the ADMT documentation catalogue:
with open(os.path.join(path2assets, "admt_documentation_catalogue.json"), "rb") as f:
    ADMT_CATALOGUE = json.load(f)['data']['catalogue']


class ArgoDocs:
    """ADMT documentation access and discovery

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
    _catalogue = ADMT_CATALOGUE

    class RIS:
        """Make a RIS file structure from a TXT file"""

        def __init__(self, file=None, fs=None):
            self.record = None
            self.fs = fs
            if file:
                self.parse(file)

        def parse(self, file):
            """Parse input file"""
            # log.debug(file)

            try:
                with self.fs.open(file, 'r', encoding="utf-8") as f:
                    TXTlines = f.readlines()
            except:  # noqa: E722
                with self.fs.open(file, 'r', encoding="latin-1") as f:
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
                    try:
                        if line[2] == " ":
                            tag = line[0:2]
                            field = line[3:]
                            # print("ok", {tag: field})
                            record[tag] = [field]
                        else:
                            # print("-", line)
                            record[tag].append(line)
                    except UnboundLocalError:
                        pass
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
            summary.append("URL: https://dx.doi.org/%s" % doc['doi'])
            summary.append("last pdf: %s" % self.pdf)
            if 'AF' in self.ris:
                summary.append("Authors: %s" % self.ris['AF'])
            if 'AB' in self.ris:
                summary.append("Abstract: %s" % self.ris['AB'])
        else:
            summary.append("- %i documents with a DOI are available in the catalogue" % len(self._catalogue))
            summary.append("> Use the method 'search' to find a document id")
            summary.append("> Use the property 'list' to check out the catalogue content")
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
                url = 'https://archimer.ifremer.fr/api/download-metadata'
                myobj = {"fields": [], "fileType": "TXT", "listId": [self.docid], "ldapList": [],
                         "exportAllAvailableFields": 'true'}
                x = requests.post(url, json=myobj)
                fs = memorystore()
                with fs.open('txt_file_content', 'w', encoding="utf-8") as f:
                    f.writelines(x.content.decode().replace('\r', '\n'))
                self._ris = self.RIS('txt_file_content', fs=fs).record
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

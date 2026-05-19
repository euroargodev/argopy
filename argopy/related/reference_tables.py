import pandas as pd
from functools import lru_cache
import collections
from pathlib import Path

from ..stores import httpstore, filestore
from ..options import OPTIONS
from ..utils import path2assets


VALID_REF = filestore(cache=True).open_json(Path(path2assets).joinpath("nvs_reference_tables.json"))['data']['valid_ref']


class ArgoNVSReferenceTables:
    """Argo Reference Tables

    Utility function to retrieve Argo Reference Tables from a NVS server.

    By default, this relies on: https://vocab.nerc.ac.uk/collection

    Examples
    --------
    Methods:

    >>> R = ArgoNVSReferenceTables()
    >>> R.search('sensor')
    >>> R.tbl(3)
    >>> R.tbl('R09')

    Properties:

    >>> R.all_tbl_name
    >>> R.all_tbl
    >>> R.valid_ref

    Notes
    -----
    This class relies on a list of valid reference table ids that is updated on every argopy release.

    """
    valid_ref = VALID_REF.copy()

    """List of all available Reference Tables"""

    def __init__(
        self,
        nvs="https://vocab.nerc.ac.uk/collection",
        cache: bool = True,
        cachedir: str = "",
    ):
        """Argo Reference Tables from NVS"""

        cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.fs = httpstore(cache=cache, cachedir=cachedir)
        self.nvs = nvs

    def _valid_ref(self, rtid):
        """
        Validate any rtid argument and return the corresponding valid ID from the list.

        Parameters
        ----------
        rtid: Input reference ID. Can be a string (e.g., "R12", "12", "r12") or a number (e.g., 12).

        Returns:
            str: Valid reference ID from the list, or None if not found.
        """
        # Convert rtid to a string and standardize its format
        if isinstance(rtid, (int, float)):
            # If rtid is a number, format it as "RXX"
            rtid_str = f"R{int(rtid):02d}"
        else:
            # If rtid is a string, convert to uppercase and standardize
            rtid_str = str(rtid).strip().upper()
            if rtid_str.startswith('R') and len(rtid_str) > 1:
                # If it starts with 'R', ensure the numeric part is two digits
                prefix = rtid_str[0]
                suffix = rtid_str[1:]
                try:
                    num = int(suffix)
                    rtid_str = f"{prefix}{num:02d}"
                except ValueError:
                    pass  # Keep the original string if conversion fails
            elif ~rtid_str.startswith('R'):
                try:
                    num = int(rtid_str)
                    rtid_str = f"R{num}"
                except ValueError:
                    pass  # Keep the original string if conversion fails

        # Check if the standardized rtid_str is in the valid_refs list
        if rtid_str in self.valid_ref:
            return rtid_str
        else:
            raise ValueError(
                f"Invalid Argo Reference Table '{rtid}', must be one in: {', '.join(self.valid_ref)}"
            )
        return rtid

    def _jsConcept2df(self, data):
        """Return all skos:Concept as class:`pandas.DataFrame`"""
        content = {
            "altLabel": [],
            "prefLabel": [],
            "definition": [],
            "deprecated": [],
            "id": [],
        }
        for k in data["@graph"]:
            if k["@type"] == "skos:Collection":
                Collection_name = k["dc:alternative"]
            elif k["@type"] == "skos:Concept":
                content["altLabel"].append(k["skos:altLabel"])
                content["prefLabel"].append(k["skos:prefLabel"]["@value"])
                content["definition"].append(k["skos:definition"]["@value"] if k["skos:definition"] != '' else None)
                content["deprecated"].append(k["owl:deprecated"])
                content["id"].append(k["@id"])
        df = pd.DataFrame.from_dict(content)
        df.name = Collection_name
        return df

    def _jsCollection(self, data):
        """Return last skos:Collection information as data"""
        for k in data["@graph"]:
            if k["@type"] == "skos:Collection":
                name = k["dc:alternative"]
                desc = k["dc:description"]
                rtid = k["@id"]
        return (name, desc, rtid)

    def get_url(self, rtid, fmt="ld+json"):
        """Return URL toward a given reference table for a given format

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12
        fmt: str, default: "ld+json"
            Format of the NVS server response. Can be: "ld+json", "rdf+xml" or "text/turtle".

        Returns
        -------
        str
        """
        rtid = self._valid_ref(rtid)
        if fmt == "ld+json":
            fmt_ext = "?_profile=nvs&_mediatype=application/ld+json"
        elif fmt == "rdf+xml":
            fmt_ext = "?_profile=nvs&_mediatype=application/rdf+xml"
        elif fmt == "text/turtle":
            fmt_ext = "?_profile=nvs&_mediatype=text/turtle"
        else:
            raise ValueError(
                "Invalid format. Must be in: 'ld+json', 'rdf+xml' or 'text/turtle'."
            )
        url = "{}/{}/current/{}".format
        return url(self.nvs, rtid, fmt_ext)

    @lru_cache
    def tbl(self, rtid):
        """Return an Argo Reference table

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12

        Returns
        -------
        class:`pandas.DataFrame`
        """
        rtid = self._valid_ref(rtid)
        js = self.fs.open_json(self.get_url(rtid))
        df = self._jsConcept2df(js)
        return df

    def tbl_name(self, rtid):
        """Return name of an Argo Reference table

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12

        Returns
        -------
        tuple('short name', 'description', 'NVS id link')
        """
        rtid = self._valid_ref(rtid)
        js = self.fs.open_json(self.get_url(rtid))
        return self._jsCollection(js)

    def search(self, txt, where="all"):
        """Search for string in tables title and/or description

        Parameters
        ----------
        txt: str
        where: str, default='all'
            Where to search, can be: 'title', 'description', 'all'

        Returns
        -------
        list of table id matching the search
        """
        results = []
        for tbl_id in self.all_tbl_name:
            title = self.tbl_name(tbl_id)[0]
            description = self.tbl_name(tbl_id)[1]
            if where == "title":
                if txt.lower() in title.lower():
                    results.append(tbl_id)
            elif where == "description":
                if txt.lower() in description.lower():
                    results.append(tbl_id)
            elif where == "all":
                if txt.lower() in description.lower() or txt.lower() in title.lower():
                    results.append(tbl_id)
        return results

    @property
    def all_tbl(self):
        """Return all Argo Reference tables

        Returns
        -------
        OrderedDict
            Dictionary with all table short names as key and table content as class:`pandas.DataFrame`
        """
        URLs = [self.get_url(rtid) for rtid in self.valid_ref]
        df_list = self.fs.open_mfjson(URLs, preprocess=self._jsConcept2df)
        all_tables = {}
        [all_tables.update({t.name: t}) for t in df_list]
        all_tables = collections.OrderedDict(sorted(all_tables.items()))
        return all_tables

    @property
    def all_tbl_name(self):
        """Return names of all Argo Reference tables

        Returns
        -------
        OrderedDict
            Dictionary with all table short names as key and table names as tuple('short name', 'description', 'NVS id link')
        """
        URLs = [self.get_url(rtid) for rtid in self.valid_ref]
        name_list = self.fs.open_mfjson(URLs, preprocess=self._jsCollection)
        all_tables = {}
        [
            all_tables.update({rtid.split("/")[-3]: (name, desc, rtid)})
            for name, desc, rtid in name_list
        ]
        all_tables = collections.OrderedDict(sorted(all_tables.items()))
        return all_tables

# -*- coding: utf-8 -*-

"""
argopy.data_fetchers.erddap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains Argo meta-data fetcher for Ifremer ERDDAP.

This is not intended to be used directly, only by the facade at fetchers.py

"""

import pandas as pd
import numpy as np
import copy
import logging
from abc import ABC, abstractmethod
from erddapy.erddapy import ERDDAP, parse_dates
from erddapy.erddapy import _quote_string_constraints as quote_string_constraints

from ..utils.format import format_oneline
from ..related import load_dict, mapp_dict
from ..stores import httpstore
from ..options import OPTIONS

log = logging.getLogger("argopy.fetchers.erddap_index")


access_points = ["wmo", "box"]
exit_formats = ["xarray", "dataframe"]
dataset_ids = ["phy"]  # First is default
api_server = "https://erddap.ifremer.fr/erddap/"  # API root url
api_server_check = (
    api_server + "/info/ArgoFloats/index.json"
)  # URL to check if the API is alive


class ErddapArgoIndexFetcher(ABC):
    """Manage access to Argo index through Ifremer ERDDAP

    ERDDAP transaction are managed with the erddapy library

    """

    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self):
        """Initialisation for a specific fetcher"""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def define_constraints(self):
        """Define erddapy constraints"""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def cname(self):
        """Return a unique string defining the request"""
        raise NotImplementedError("Not implemented")

    ###
    # Methods that must not change
    ###
    def __init__(self, cache: bool = False, cachedir: str = "", **kwargs):
        """Instantiate an ERDDAP Argo index loader"""
        # if version.parse(fsspec.__version__) > version.parse("0.8.3") and cache:
        #     log.warning("Caching not available for WMO access point, falls back on NO cache "
        #                 "(http cache store not compatible with erddap wmo requests)")
        #     cache = False
        if cache:
            log.warning(
                "Caching not available for WMO access point, falls back on NO cache "
                "(http cache store not compatible with erddap wmo requests)"
            )
            cache = False
        self.fs = httpstore(
            cache=cache, cachedir=cachedir, timeout=OPTIONS["api_timeout"]
        )
        self.definition = "Ifremer erddap Argo index fetcher"
        self.dataset_id = "index"
        self.server = api_server
        self.init(**kwargs)
        self._init_erddapy()

    def __repr__(self):
        summary = ["<indexfetcher.erddap>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % api_server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def _format(self, x, typ):
        """String formatting helper"""
        if typ == "lon":
            if x < 0:
                x = 360.0 + x
            return ("%05d") % (x * 100.0)
        if typ == "lat":
            return ("%05d") % (x * 100.0)
        if typ == "prs":
            return ("%05d") % (np.abs(x) * 10.0)
        if typ == "tim":
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        return str(x)

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(server=self.server, protocol="tabledap")
        self.erddap.response = "csv"
        self.erddap.dataset_id = "ArgoFloats-index"
        return self

    @property
    def cachepath(self):
        """Return path to cache file for this request"""
        return self.fs.cachepath(self.url)

    @property
    def url(self, response=None):
        """Return the URL used to download data"""
        # Replace erddapy get_download_url
        # We need to replace it to better handle http responses with by-passing the _check_url_response
        # https://github.com/ioos/erddapy/blob/fa1f2c15304938cd0aa132946c22b0427fd61c81/erddapy/erddapy.py#L247

        # Define constraint to select this box of data:
        self.define_constraints()  # This will affect self.erddap.constraints

        # Define the list of variables to retrieve - all for the index
        self.erddap.variables = [
            "file",
            "date",
            "longitude",
            "latitude",
            "ocean",
            "profiler_type",
            "institution",
            "date_update",
        ]

        #
        dataset_id = self.erddap.dataset_id
        protocol = self.erddap.protocol
        variables = self.erddap.variables
        if not response:
            response = self.erddap.response
        constraints = self.erddap.constraints
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"
        if variables:
            variables = ",".join(variables)
            url += f"{variables}"

        if constraints:
            _constraints = copy.copy(constraints)
            for k, v in _constraints.items():
                if k.startswith("time"):
                    _constraints.update({k: parse_dates(v)})
            _constraints = quote_string_constraints(_constraints)
            _constraints = "".join([f"&{k}{v}" for k, v in _constraints.items()])

            url += f"{_constraints}"

        url += '&distinct()&orderBy("date")'
        # In erddapy:
        # url = _distinct(url, **kwargs)
        # return _check_url_response(url, **self.requests_kwargs)
        return url

    def to_dataframe(self):
        """Load Argo index and return a pandas dataframe"""

        # Download data: get a csv, open it as pandas dataframe, create wmo field
        df = self.fs.read_csv(self.url, parse_dates=True, skiprows=[1])

        # erddap date format : 2019-03-21T00:00:35Z
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%dT%H:%M:%SZ")
        df["date_update"] = pd.to_datetime(
            df["date_update"], format="%Y-%m-%dT%H:%M:%SZ"
        )
        df["wmo"] = df.file.apply(lambda x: int(x.split("/")[1]))

        # institution & profiler mapping
        institution_dictionnary = load_dict("institutions")
        df["tmp1"] = df.institution.apply(
            lambda x: mapp_dict(institution_dictionnary, x)
        )
        df = df.rename(
            columns={"institution": "institution_code", "tmp1": "institution"}
        )

        profiler_dictionnary = load_dict("profilers")
        df["profiler"] = df.profiler_type.apply(
            lambda x: mapp_dict(profiler_dictionnary, int(x))
        )
        df = df.rename(columns={"profiler_type": "profiler_code"})

        return df

    def to_xarray(self):
        """Load Argo index and return a xarray Dataset"""
        return self.to_dataframe().to_xarray()

    def clear_cache(self):
        """Remove cache files and entries from resources open with this fetcher"""
        return self.fs.clear_cache()


class Fetch_wmo(ErddapArgoIndexFetcher):
    """Manage access to Argo Index through Ifremer ERDDAP for: a list of WMOs"""

    access_point = "wmo"

    def init(self, WMO=[], **kwargs):
        """Create Argo data loader for WMOs

        Parameters
        ----------
        WMO : list(int)
            The list of WMOs to load all Argo data for.
        """
        self.WMO = WMO
        self.definition = "Ifremer erddap Argo Index fetcher for floats"
        return self

    def define_constraints(self):
        """Define erddap constraints"""
        #  'file=~': "(.*)(R|D)(6902746_|6902747_)(.*)"
        self.erddap.constraints = {
            "file=~": "(.*)(R|D)(" + "|".join(["%i" % i for i in self.WMO]) + ")(_.*)"
        }
        return self

    def cname(self):
        """Return a unique string defining the constraints"""
        if len(self.WMO) > 1:
            listname = ["WMO%i" % i for i in self.WMO]
            listname = ";".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
        return listname


class Fetch_box(ErddapArgoIndexFetcher):
    """Manage access to Argo Index through Ifremer ERDDAP for: an ocean rectangle"""

    access_point = "box"

    def init(self, box=[], **kwargs):
        """Create Argo Index loader

        Parameters
        ----------
        box : list(float, float, float, float, str, str)
        The box domain to load all Argo data for:
        box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        self.BOX = box.copy()
        if len(box) == 4:
            # Use all time line:
            self.BOX.append("1900-01-01")
            self.BOX.append("2100-12-31")
        self.definition = "Ifremer erddap Argo Index fetcher for a space/time region"

        return self

    def define_constraints(self):
        """Define request constraints"""
        self.erddap.constraints = {"longitude>=": self.BOX[0]}
        self.erddap.constraints.update({"longitude<=": self.BOX[1]})
        self.erddap.constraints.update({"latitude>=": self.BOX[2]})
        self.erddap.constraints.update({"latitude<=": self.BOX[3]})
        self.erddap.constraints.update({"date>=": self.BOX[4]})
        self.erddap.constraints.update({"date<=": self.BOX[5]})
        return None

    def cname(self):
        """Return a unique string defining the constraints"""
        BOX = self.BOX
        boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % (
            BOX[0],
            BOX[1],
            BOX[2],
            BOX[3],
            self._format(BOX[4], "tim"),
            self._format(BOX[5], "tim"),
        )
        return boxname

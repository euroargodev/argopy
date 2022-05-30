#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for Argovis.
#

import numpy as np
import pandas as pd
import getpass
import logging
from .proto import ArgoDataFetcherProto
from abc import abstractmethod
import warnings

from argopy.stores import httpstore
from argopy.options import OPTIONS
from argopy.utilities import list_standard_variables, format_oneline, Chunker
from argopy.errors import DataNotFound


access_points = ['wmo', 'box']
exit_formats = ['xarray']
dataset_ids = ['phy']  # First is default
api_server = 'https://argovis.colorado.edu'  # API root url
api_server_check = (
    api_server + "/selection/overview"
)  # URL to check if the API is alive

log = logging.getLogger("argopy.argovis.data")


class ArgovisDataFetcher(ArgoDataFetcherProto):
    ###
    # Methods to be customised for a specific Argovis request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """ Initialisation for a specific fetcher """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri(self):
        """ Return the URL used to download data """
        raise NotImplementedError("Not implemented")

    ###
    # Methods that must not change
    ###
    def __init__(
        self,
        ds: str = "",
        cache: bool = False,
        cachedir: str = "",
        parallel: bool = False,
        parallel_method: str = "thread",
        progress: bool = False,
        chunks: str = "auto",
        chunks_maxsize: dict = {},
        api_timeout: int = 0,
        **kwargs
    ):
        """ Instantiate an Argovis Argo data loader

        Parameters
        ----------
        ds: str (optional)
            Dataset to load: 'phy' or 'bgc'
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        parallel: bool (optional)
            Chunk request to use parallel fetching (default: False)
        parallel_method: str (optional)
            Define the parallelization method: ``thread``, ``process`` or a :class:`dask.distributed.client.Client`.
        progress: bool (optional)
            Show a progress bar or not when ``parallel`` is set to True.
        chunks: 'auto' or dict of integers (optional)
            Dictionary with request access point as keys and number of chunks to create as values.
            Eg: {'wmo': 10} will create a maximum of 10 chunks along WMOs when used with ``Fetch_wmo``.
        chunks_maxsize: dict (optional)
            Dictionary with request access point as keys and chunk size as values (used as maximum values in
            'auto' chunking).
            Eg: {'wmo': 5} will create chunks with as many as 5 WMOs each.
        api_timeout: int (optional)
            Argovis API request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout)
        self.definition = "Argovis Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = api_server

        if not isinstance(parallel, bool):
            parallel_method = parallel
            parallel = True
        if parallel_method not in ["thread"]:
            raise ValueError("argovis only support multi-threading, use 'thread' instead of '%s'" % parallel_method)
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

        self.init(**kwargs)
        self.key_map = {
            "date": "TIME",
            "date_qc": "TIME_QC",
            "lat": "LATITUDE",
            "lon": "LONGITUDE",
            "cycle_number": "CYCLE_NUMBER",
            "DATA_MODE": "DATA_MODE",
            "DIRECTION": "DIRECTION",
            "platform_number": "PLATFORM_NUMBER",
            "position_qc": "POSITION_QC",
            "pres": "PRES",
            "temp": "TEMP",
            "psal": "PSAL",
            "index": "N_POINTS",
        }

    def __repr__(self):
        summary = ["<datafetcher.argovis>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % api_server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def _add_history(self, this, txt):
        if "history" in this.attrs:
            this.attrs["history"] += "; %s" % txt
        else:
            this.attrs["history"] = txt
        return this

    def _add_attributes(self, this):  # noqa: C901
        """ Add variables attributes not return by argovis requests

            #todo: This is hard coded, but should be retrieved from an API somewhere
        """
        for v in this.data_vars:
            if "TEMP" in v and "_QC" not in v:
                this[v].attrs = {
                    "long_name": "SEA TEMPERATURE IN SITU ITS-90 SCALE",
                    "standard_name": "sea_water_temperature",
                    "units": "degree_Celsius",
                    "valid_min": -2.0,
                    "valid_max": 40.0,
                    "resolution": 0.001,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            if "PSAL" in v and "_QC" not in v:
                this[v].attrs = {
                    "long_name": "PRACTICAL SALINITY",
                    "standard_name": "sea_water_salinity",
                    "units": "psu",
                    "valid_min": 0.0,
                    "valid_max": 43.0,
                    "resolution": 0.001,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            if "PRES" in v and "_QC" not in v:
                this[v].attrs = {
                    "long_name": "Sea Pressure",
                    "standard_name": "sea_water_pressure",
                    "units": "decibar",
                    "valid_min": 0.0,
                    "valid_max": 12000.0,
                    "resolution": 0.1,
                    "axis": "Z",
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            if "DOXY" in v and "_QC" not in v:
                this[v].attrs = {
                    "long_name": "Dissolved oxygen",
                    "standard_name": "moles_of_oxygen_per_unit_mass_in_sea_water",
                    "units": "micromole/kg",
                    "valid_min": -5.0,
                    "valid_max": 600.0,
                    "resolution": 0.001,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            if "_QC" in v:
                attrs = {
                    "long_name": "Global quality flag of %s profile" % v,
                    "convention": "Argo reference table 2a",
                }
                this[v].attrs = attrs

        if "CYCLE_NUMBER" in this.data_vars:
            this["CYCLE_NUMBER"].attrs = {
                "long_name": "Float cycle number",
                "convention": "0..N, 0 : launch cycle (if exists), 1 : first complete cycle",
            }

        if "DATA_MODE" in this.data_vars:
            this["DATA_MODE"].attrs = {
                "long_name": "Delayed mode or real time data",
                "convention": "R : real time; D : delayed mode; A : real time with adjustment",
            }

        if "DIRECTION" in this.data_vars:
            this["DIRECTION"].attrs = {
                "long_name": "Direction of the station profiles",
                "convention": "A: ascending profiles, D: descending profiles",
            }

        if "PLATFORM_NUMBER" in this.data_vars:
            this["PLATFORM_NUMBER"].attrs = {
                "long_name": "Float unique identifier",
                "convention": "WMO float identifier : A9IIIII",
            }

        return this

    @property
    def cachepath(self):
        """ Return path to cache file for this request """
        return [self.fs.cachepath(url) for url in self.uri]

    def cname(self):
        """ Return a unique string defining the constraints """
        return self._cname()

    def url_encode(self, urls):
        """ Return safely encoded list of urls

            This was made to debug for fsspec caching system not working with cache of profile and region in argovis
            Not working yet, see: https://github.com/euroargodev/argopy/issues/101
        """
        # return urls
        def safe_for_fsspec_cache(url):
            url = url.replace("[", "%5B")  # This is the one really necessary
            url = url.replace("]", "%5D")  # For consistency
            return url
        return [safe_for_fsspec_cache(url) for url in urls]
        # return [urllib.parse.quote(url, safe='/:?=[]&') for url in urls]

    def json2dataframe(self, profiles):
        """ convert json data to Pandas DataFrame """
        # Make sure we deal with a list
        if isinstance(profiles, list):
            data = profiles
        else:
            data = [profiles]
        # Transform
        rows = []
        for profile in data:
            keys = [x for x in profile.keys() if x not in ["measurements", "bgcMeas"]]
            meta_row = dict((key, profile[key]) for key in keys)
            meta_row['date'] = meta_row['date'][0:-2] if meta_row['date'][-1] == 'Z' else meta_row['date']  # Remove timezone #101
            for row in profile["measurements"]:
                row.update(meta_row)
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def to_dataframe(self, errors: str = 'ignore'):
        """ Load Argo data and return a Pandas dataframe """

        # Download data:
        if not self.parallel:
            method = "sequential"
        else:
            method = self.parallel_method
        df_list = self.fs.open_mfjson(
            self.uri, method=method, preprocess=self.json2dataframe, progress=self.progress, errors=errors
        )

        # Merge results (list of dataframe):
        for i, df in enumerate(df_list):
            df = df.reset_index()
            df = df.rename(columns=self.key_map)
            df = df[[value for value in self.key_map.values() if value in df.columns]]
            df_list[i] = df
        df = pd.concat(df_list, ignore_index=True)
        if df.shape[0] == 0:
            raise DataNotFound("No data found for: %s" % self.cname())
        df.sort_values(by=["TIME", "PRES"], inplace=True)
        df = df.set_index(["N_POINTS"])
        return df

    def to_xarray(self, errors: str = 'ignore'):
        """ Download and return data as xarray Datasets """
        ds = self.to_dataframe(errors=errors).to_xarray()
        ds = ds.sortby(
            ["TIME", "PRES"]
        )  # should already be sorted by date in descending order
        ds["N_POINTS"] = np.arange(
            0, len(ds["N_POINTS"])
        )  # Re-index to avoid duplicate values

        # Set coordinates:
        # ds = ds.set_coords('N_POINTS')
        coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
        ds = ds.reset_coords()
        ds["N_POINTS"] = ds["N_POINTS"]
        # Convert all coordinate variable names to upper case
        for v in ds.data_vars:
            ds = ds.rename({v: v.upper()})
        ds = ds.set_coords(coords)

        # Cast data types and add variable attributes (not available in the csv download):
        ds['TIME'] = ds['TIME'].astype(np.datetime64)
        ds = self._add_attributes(ds)
        ds = ds.argo.cast_types()

        # Remove argovis file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        elif self.dataset_id == "ref":
            ds.attrs["DATA_ID"] = "ARGO_Reference"
        elif self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.server
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = self.uri
        ds = ds[np.sort(ds.data_vars)]
        ds = self.filter_domain(ds)  # https://github.com/euroargodev/argopy/issues/48
        return ds

    def filter_data_mode(self, ds, **kwargs):
        # Argovis data already curated !
        # ds = ds.argo.filter_data_mode(errors='ignore', **kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_qc(self, ds, **kwargs):
        # Argovis data already curated !
        # ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_variables(self, ds, mode="standard"):
        if mode == "standard":
            to_remove = sorted(
                list(set(list(ds.data_vars)) - set(list_standard_variables()))
            )
            return ds.drop_vars(to_remove)
        else:
            return ds

    def filter_domain(self, ds):
        """ Enforce rectangular box shape

        This is a temporary fix for https://github.com/euroargodev/argopy/issues/48
        """
        if hasattr(self, 'BOX'):
            ds = ds.where(ds['LATITUDE'] >= self.BOX[2], drop=True)
            ds = ds.where(ds['LATITUDE'] <= self.BOX[3], drop=True)
            return ds
        else:
            return ds


class Fetch_wmo(ArgovisDataFetcher):
    def init(self, WMO=[], CYC=None, **kwargs):
        """ Create Argo data loader for WMOs and CYCs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
            CYC : int, np.array(int), list(int)
                The cycle numbers to load.
        """
        self.WMO = WMO
        self.CYC = CYC

        self.definition = "?"
        if self.dataset_id == "phy":
            self.definition = "Argovis Argo data fetcher for floats"
        return self

    def get_url(self, wmo: int, cyc: int = None) -> str:
        """ Return path toward the source file of a given wmo/cyc pair """
        if cyc is None:
            return (self.server + "/catalog/platforms/{}").format(str(wmo))
        else:
            profIds = [str(wmo) + "_" + str(c) for c in cyc]
            return (
                (self.server + "/catalog/mprofiles/?ids={}")
                .format(profIds)
                .replace(" ", "")
            )

    @property
    def uri(self):
        """ List of URLs to load for a request

        Returns
        -------
        list(str)
        """
        def list_bunch(wmos, cycs):
            this = []
            for wmo in wmos:
                if cycs is None:
                    this.append(self.get_url(wmo))
                else:
                    this.append(self.get_url(wmo, cycs))
            return this

        urls = list_bunch(self.WMO, self.CYC)
        return self.url_encode(urls)


class Fetch_box(ArgovisDataFetcher):

    def init(self, box: list, **kwargs):
        """ Create Argo data loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
                or:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        self.BOX = box.copy()
        if len(self.BOX) == 6:
            # Select the last months of data:
            end = pd.to_datetime("now", utc=True)
            start = end - pd.DateOffset(months=1)
            self.BOX.append(start.strftime("%Y-%m-%d"))
            self.BOX.append(end.strftime("%Y-%m-%d"))

        self.definition = "?"
        if self.dataset_id == "phy":
            self.definition = "Argovis Argo data fetcher for a space/time region"
        return self

    def get_url_shape(self):
        """ Return the URL used to download data """
        shape = [
            [
                [self.BOX[0], self.BOX[2]],  # ll
                [self.BOX[0], self.BOX[3]],  # ul
                [self.BOX[1], self.BOX[3]],  # ur
                [self.BOX[1], self.BOX[2]],  # lr
                [self.BOX[0], self.BOX[2]],  # ll
            ]
        ]
        strShape = str(shape).replace(" ", "")
        url = self.server + "/selection/profiles"
        url += "?startDate={}".format(
            pd.to_datetime(self.BOX[6]).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        url += "&endDate={}".format(
            pd.to_datetime(self.BOX[7]).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        url += "&shape={}".format(strShape)
        url += "&presRange=[{},{}]".format(self.BOX[4], self.BOX[5])
        return url

    def get_url_rect(self):
        """ Return the URL used to download data """
        def strCorner(b, i): return str([b[i[0]], b[i[1]]]).replace(" ", "")
        def strDate(b, i): return pd.to_datetime(b[i]).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = self.server + "/selection/box/profiles"
        url += "?startDate={}".format(strDate(self.BOX, 6))
        url += "&endDate={}".format(strDate(self.BOX, 7))
        url += "&presRange=[{},{}]".format(self.BOX[4], self.BOX[5])
        url += "&llCorner={}".format(strCorner(self.BOX, [0, 2]))
        url += "&urCorner={}".format(strCorner(self.BOX, [1, 3]))
        return url

    def get_url(self):
        return self.get_url_shape()
        # return self.get_url_rect()

    @property
    def uri(self):
        """ List of URLs to load for a request

        Returns
        -------
        list(str)
        """
        Lt = np.timedelta64(pd.to_datetime(self.BOX[7]) - pd.to_datetime(self.BOX[6]), "D")
        MaxLenTime = 90
        MaxLen = np.timedelta64(MaxLenTime, "D")

        urls = []
        if not self.parallel:
            # Check if the time range is not larger than allowed (90 days):
            if Lt > MaxLen:
                self.Chunker = Chunker(
                    {"box": self.BOX},
                    chunks={"lon": 1, "lat": 1, "dpt": 1, "time": "auto"},
                    chunksize={"time": MaxLenTime},
                )
                boxes = self.Chunker.fit_transform()
                for box in boxes:
                    urls.append(Fetch_box(box=box, ds=self.dataset_id).get_url())
            else:
                urls.append(self.get_url())
        else:
            if 'time' not in self.chunks_maxsize:
                self.chunks_maxsize['time'] = MaxLenTime
            elif self.chunks_maxsize['time'] > MaxLenTime:
                warnings.warn(("argovis only allows requests of %i days interval, "
                               "modify chunks_maxsize['time'] to %i" % (MaxLenTime, MaxLenTime)))
                self.chunks_maxsize['time'] = MaxLenTime

            # Ensure time chunks will never exceed what's allowed by argovis:
            if Lt > MaxLen and 'time' in self.chunks and self.chunks['time'] not in ['auto']:
                self.chunks['time'] = 'auto'

            self.Chunker = Chunker(
                {"box": self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize
            )
            boxes = self.Chunker.fit_transform()
            for box in boxes:
                urls.append(Fetch_box(box=box, ds=self.dataset_id).get_url())

        return self.url_encode(urls)

    @property
    def url(self):
        return self.get_url_shape()
        # return self.get_url_rect()

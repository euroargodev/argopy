# -*- coding: utf-8 -*-

"""
argopy.data_fetchers.erddap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains Argo data fetcher for Ifremer ERDDAP.

This is not intended to be used directly, only by the facade at fetchers.py

"""

import pandas as pd
import numpy as np
import copy

from abc import abstractmethod
import getpass

from .proto import ArgoDataFetcherProto
from argopy.options import OPTIONS
from argopy.utilities import list_standard_variables, Chunker, format_oneline
from argopy.stores import httpstore
from ..errors import ErddapServerError
from aiohttp import ClientResponseError


# Load erddapy according to available version (breaking changes in v0.8.0)
try:
    from erddapy import ERDDAP
    from erddapy.utilities import parse_dates, quote_string_constraints
except:  # noqa: E722
    # >= v0.8.0
    from erddapy.erddapy import ERDDAP
    from erddapy.erddapy import _quote_string_constraints as quote_string_constraints
    from erddapy.erddapy import parse_dates


access_points = ['wmo', 'box']
exit_formats = ['xarray']
dataset_ids = ['phy', 'ref', 'bgc']  # First is default
api_server = 'https://erddap.ifremer.fr/erddap/'  # API root url
api_server_check = api_server + '/info/ArgoFloats/index.json'  # URL to check if the API is alive


class ErddapArgoDataFetcher(ArgoDataFetcherProto):
    """ Manage access to Argo data through Ifremer ERDDAP

        ERDDAP transaction are managed with the erddapy library

        This class is a prototype not meant to be instantiated directly

    """

    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """ Initialisation for a specific fetcher """
        raise NotImplementedError("ErddapArgoDataFetcher.init not implemented")

    @abstractmethod
    def define_constraints(self):
        """ Define erddapy constraints """
        raise NotImplementedError("ErddapArgoDataFetcher.define_constraints not implemented")

    @property
    @abstractmethod
    def uri(self) -> list:
        """ Return the list of Unique Resource Identifier (URI) to download data """
        raise NotImplementedError("ErddapArgoDataFetcher.uri not implemented")

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
        **kwargs,
    ):
        """ Instantiate an ERDDAP Argo data fetcher

        Parameters
        ----------
        ds: str (optional)
            Dataset to load: 'phy' or 'ref' or 'bgc'
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
            Erddap request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')
        self.definition = "Ifremer erddap Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = api_server

        if not isinstance(parallel, bool):
            parallel_method = parallel
            parallel = True
        if parallel_method not in ["thread"]:
            raise ValueError(
                "erddap only support multi-threading, use 'thread' instead of '%s'"
                % parallel_method
            )
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

        self.init(**kwargs)
        self._init_erddapy()

    def __repr__(self):
        summary = ["<datafetcher.erddap>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % api_server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def _add_attributes(self, this):  # noqa: C901
        """ Add variables attributes not return by erddap requests (csv)

            This is hard coded, but should be retrieved from an API somewhere
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

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(server=self.server, protocol="tabledap")
        self.erddap.response = (
            "nc"  # This is a major change in v0.4, we used to work with csv files
        )

        if self.dataset_id == "phy":
            self.erddap.dataset_id = "ArgoFloats"
        elif self.dataset_id == "ref":
            self.erddap.dataset_id = "ArgoFloats-ref"
        elif self.dataset_id == "bgc":
            self.erddap.dataset_id = "ArgoFloats-synthetic-BGC"
        elif self.dataset_id == "fail":
            self.erddap.dataset_id = "invalid_db"
        else:
            raise ValueError(
                "Invalid database short name for Ifremer erddap (use: 'phy', 'bgc' or 'ref')"
            )
        return self

    @property
    def _minimal_vlist(self):
        """ Return the minimal list of variables to retrieve measurements for """
        vlist = list()
        if self.dataset_id == "phy":
            plist = [
                "data_mode",
                "latitude",
                "longitude",
                "position_qc",
                "time",
                "time_qc",
                "direction",
                "platform_number",
                "cycle_number",
                "config_mission_number",
                "vertical_sampling_scheme",
            ]
            [vlist.append(p) for p in plist]

            plist = ["pres", "temp", "psal"]
            [vlist.append(p) for p in plist]
            [vlist.append(p + "_qc") for p in plist]
            [vlist.append(p + "_adjusted") for p in plist]
            [vlist.append(p + "_adjusted_qc") for p in plist]
            [vlist.append(p + "_adjusted_error") for p in plist]

        if self.dataset_id == "bgc":
            plist = [
                "parameter_data_mode",
                "latitude",
                "longitude",
                "position_qc",
                "time",
                "time_qc",
                "direction",
                "platform_number",
                "cycle_number",
                "config_mission_number",
            ]
            [vlist.append(p) for p in plist]

            plist = ["pres", "temp", "psal",
                     "cndc",
                     "doxy",
                     "beta_backscattering",
                     "fluorescence_chla",
                     # "fluorescence_cdom",
                     # "side_scattering_turbidity",
                     # "transmittance_particle_beam_attenuation",
                     "bbp",
                     "turbidity",
                     "cp",
                     "chla",
                     "cdom",
                     "nitrate",
                     ]
            [vlist.append(p) for p in plist]
            [vlist.append(p + "_qc") for p in plist]
            [vlist.append(p + "_adjusted") for p in plist]
            [vlist.append(p + "_adjusted_qc") for p in plist]
            [vlist.append(p + "_adjusted_error") for p in plist]

        elif self.dataset_id == "ref":
            plist = ["latitude", "longitude", "time", "platform_number", "cycle_number"]
            [vlist.append(p) for p in plist]
            plist = ["pres", "temp", "psal", "ptmp"]
            [vlist.append(p) for p in plist]

        return vlist

    @property
    def _dtype(self):
        """ Return a dictionary of data types for each variable requested to erddap in the minimal vlist """
        dref = {
            "data_mode": object,
            "latitude": np.float64,
            "longitude": np.float64,
            "position_qc": np.int64,
            "time": object,
            "time_qc": np.int64,
            "direction": object,
            "platform_number": np.int64,
            "config_mission_number": np.int64,
            "vertical_sampling_scheme": object,
            "cycle_number": np.int64,
            "pres": np.float64,
            "temp": np.float64,
            "psal": np.float64,
            "doxy": np.float64,
            "pres_qc": np.int64,
            "temp_qc": object,
            "psal_qc": object,
            "doxy_qc": object,
            "pres_adjusted": np.float64,
            "temp_adjusted": np.float64,
            "psal_adjusted": np.float64,
            "doxy_adjusted": np.float64,
            "pres_adjusted_qc": object,
            "temp_adjusted_qc": object,
            "psal_adjusted_qc": object,
            "doxy_adjusted_qc": object,
            "pres_adjusted_error": np.float64,
            "temp_adjusted_error": np.float64,
            "psal_adjusted_error": np.float64,
            "doxy_adjusted_error": np.float64,
            "ptmp": np.float64,
        }
        plist = self._minimal_vlist
        response = {}
        for p in plist:
            if p in dref:
                response[p] = dref[p]
            else:
                response[p] = object
        return response

    def cname(self):
        """ Return a unique string defining the constraints """
        return self._cname()

    @property
    def cachepath(self):
        """ Return path to cached file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(uri) for uri in self.uri]

    def get_url(self):
        """ Return the URL to download data requested

        Returns
        -------
        str
        """
        # Replace erddapy get_download_url()
        # We need to replace it to better handle http responses with by-passing the _check_url_response
        # https://github.com/ioos/erddapy/blob/fa1f2c15304938cd0aa132946c22b0427fd61c81/erddapy/erddapy.py#L247

        # First part of the URL:
        protocol = self.erddap.protocol
        dataset_id = self.erddap.dataset_id
        response = self.erddap.response
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"

        # Add variables to retrieve:
        self.erddap.variables = (
            self._minimal_vlist
        )  # Define the list of variables to retrieve
        variables = self.erddap.variables
        variables = ",".join(variables)
        url += f"{variables}"

        # Add constraints:
        self.define_constraints()  # Define constraint to select this box of data (affect self.erddap.constraints)
        constraints = self.erddap.constraints
        _constraints = copy.copy(constraints)
        for k, v in _constraints.items():
            if k.startswith("time"):
                _constraints.update({k: parse_dates(v)})
        _constraints = quote_string_constraints(_constraints)
        _constraints = "".join([f"&{k}{v}" for k, v in _constraints.items()])
        url += f"{_constraints}"

        # Last part:
        url += '&distinct()&orderBy("time,pres")'
        return url

    @property
    def N_POINTS(self) -> int:
        """ Number of measurements expected to be returned by a request """
        try:
            url = self.get_url().replace("." + self.erddap.response, ".ncHeader")
            with self.fs.open(url) as of:
                ncHeader = of.read().decode("utf-8")
            lines = [line for line in ncHeader.splitlines() if "row = " in line][0]
            return int(lines.split("=")[1].split(";")[0])
        except Exception:
            raise ErddapServerError("Erddap server can't return ncHeader for this url. ")

    def to_xarray(self, errors: str = 'ignore'):
        """ Load Argo data and return a xarray.DataSet """

        # Download data
        if not self.parallel:
            if len(self.uri) == 1:
                try:
                    ds = self.fs.open_dataset(self.uri[0])
                except ClientResponseError as e:
                    raise ErddapServerError(e.message)
            else:
                try:
                    ds = self.fs.open_mfdataset(
                        self.uri, method="sequential", progress=self.progress, errors=errors
                    )
                except ClientResponseError as e:
                    raise ErddapServerError(e.message)
        else:
            try:
                ds = self.fs.open_mfdataset(
                    self.uri, method=self.parallel_method, progress=self.progress, errors=errors
                )
            except ClientResponseError as e:
                raise ErddapServerError(e.message)

        ds = ds.rename({"row": "N_POINTS"})

        # Post-process the xarray.DataSet:

        # Set coordinates:
        coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
        ds = ds.reset_coords()
        ds["N_POINTS"] = ds["N_POINTS"]
        # Convert all coordinate variable names to upper case
        for v in ds.data_vars:
            ds = ds.rename({v: v.upper()})
        ds = ds.set_coords(coords)

        # Cast data types and add variable attributes (not available in the csv download):
        ds = self._add_attributes(ds)
        ds = ds.argo.cast_types()

        # More convention:
        #         ds = ds.rename({'pres': 'pressure'})

        # Remove erddap file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        elif self.dataset_id == "ref":
            ds.attrs["DATA_ID"] = "ARGO_Reference"
        elif self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.erddap.server
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = self.uri
        ds = ds[np.sort(ds.data_vars)]

        #
        return ds

    def filter_data_mode(self, ds, **kwargs):
        ds = ds.argo.filter_data_mode(errors="ignore", **kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_qc(self, ds, **kwargs):
        ds = ds.argo.filter_qc(**kwargs)
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


class Fetch_wmo(ErddapArgoDataFetcher):
    """ Manage access to Argo data through Ifremer ERDDAP for: a list of WMOs

    This class is instantiated when a call is made to these facade access points:
        - `ArgoDataFetcher(src='erddap').float(**)`
        - `ArgoDataFetcher(src='erddap').profile(**)`

    """

    def init(self, WMO=[], CYC=None, **kw):
        """ Create Argo data loader for WMOs

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
            self.definition = "Ifremer erddap Argo data fetcher for floats"
        elif self.dataset_id == "ref":
            self.definition = "Ifremer erddap Argo REFERENCE data fetcher for floats"
        return self

    def define_constraints(self):
        """ Define erddap constraints """
        self.erddap.constraints = {
            "platform_number=~": "|".join(["%i" % i for i in self.WMO])
        }
        if self.CYC is not None:
            self.erddap.constraints.update(
                {"cycle_number=~": "|".join(["%i" % i for i in self.CYC])}
            )
        return self

    @property
    def uri(self):
        """ List of URLs to load for a request

        Returns
        -------
        list(str)
        """
        if not self.parallel:
            chunks = "auto"
            chunks_maxsize = {'wmo': 5}
        else:
            chunks = self.chunks
            chunks_maxsize = self.chunks_maxsize
        self.Chunker = Chunker(
            {"wmo": self.WMO}, chunks=chunks, chunksize=chunks_maxsize
        )
        wmo_grps = self.Chunker.fit_transform()
        urls = []
        for wmos in wmo_grps:
            urls.append(
                Fetch_wmo(
                    WMO=wmos, CYC=self.CYC, ds=self.dataset_id, parallel=False
                ).get_url()
            )
        return urls


class Fetch_box(ErddapArgoDataFetcher):
    """ Manage access to Argo data through Ifremer ERDDAP for: an ocean rectangle
    """

    def init(self, box: list, **kw):
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

        if self.dataset_id == "phy":
            self.definition = "Ifremer erddap Argo data fetcher for a space/time region"
        elif self.dataset_id == "ref":
            self.definition = (
                "Ifremer erddap Argo REFERENCE data fetcher for a space/time region"
            )

        return self

    def define_constraints(self):
        """ Define request constraints """
        self.erddap.constraints = {"longitude>=": self.BOX[0]}
        self.erddap.constraints.update({"longitude<=": self.BOX[1]})
        self.erddap.constraints.update({"latitude>=": self.BOX[2]})
        self.erddap.constraints.update({"latitude<=": self.BOX[3]})
        self.erddap.constraints.update({"pres>=": self.BOX[4]})
        self.erddap.constraints.update({"pres<=": self.BOX[5]})
        if len(self.BOX) == 8:
            self.erddap.constraints.update({"time>=": self.BOX[6]})
            self.erddap.constraints.update({"time<=": self.BOX[7]})
        return None

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        if not self.parallel:
            return [self.get_url()]
        else:
            self.Chunker = Chunker(
                {"box": self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize
            )
            boxes = self.Chunker.fit_transform()
            urls = []
            for box in boxes:
                urls.append(Fetch_box(box=box, ds=self.dataset_id).get_url())
            return urls

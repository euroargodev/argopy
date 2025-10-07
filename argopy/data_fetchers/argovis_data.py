import numpy as np
import pandas as pd
import xarray as xr
import getpass
import logging
from abc import abstractmethod
import warnings

from ..stores import httpstore
from ..options import OPTIONS, DEFAULT, PARALLEL_SETUP
from ..utils.chunking import Chunker
from ..errors import DataNotFound
from .. import __version__
from .proto import ArgoDataFetcherProto
from .argovis_data_processors import pre_process, add_attributes

access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy"]  # First is default
api_server = "https://argovis-api.colorado.edu"
api_server_check = "https://argovis-api.colorado.edu/ping"

log = logging.getLogger("argopy.argovis.data")


class ArgovisDataFetcher(ArgoDataFetcherProto):
    data_source = "argovis"

    ###
    # Methods to be customised for a specific Argovis request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """Initialisation for a specific fetcher"""
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri(self):
        """Return the URL used to download data"""
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
        progress: bool = False,
        chunks: str = "auto",
        chunks_maxsize: dict = {},
        api_timeout: int = 0,
        **kwargs,
    ):
        """Instantiate an Argovis Argo data loader

        Parameters
        ----------
        ds: str (optional)
            Dataset to load: 'phy' or 'bgc'
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        parallel: bool, str, :class:`distributed.Client`, default: False
            Set whether to use parallelization or not, and possibly which method to use.

                Possible values:
                    - ``False``: no parallelization is used
                    - ``True``: use default method specified by the ``parallel_default_method`` option
                    - any other values accepted by the ``parallel_default_method`` option
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
        self.definition = "Argovis Argo data fetcher"
        self.dataset_id = OPTIONS["ds"] if ds == "" else ds
        self.user_mode = kwargs["mode"] if "mode" in kwargs else OPTIONS["mode"]
        self.server = kwargs["server"] if "server" in kwargs else api_server
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.store_opts = {
            "cache": cache,
            "cachedir": cachedir,
            "timeout": timeout,
            # "size_policy": "head",  # deprecated
            "client_kwargs": {"headers": {"x-argokey": OPTIONS["argovis_api_key"], "Argopy-Version": __version__}},
        }
        self.fs = kwargs["fs"] if "fs" in kwargs else httpstore(**self.store_opts)

        self.parallelize, self.parallel_method = PARALLEL_SETUP(parallel)

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
        summary.append(self._repr_data_source)
        summary.append(self._repr_access_point)
        summary.append(self._repr_server)
        api_key = self.fs.fs.client_kwargs["headers"]["x-argokey"]
        if api_key == DEFAULT["argovis_api_key"]:
            summary.append(
                "🗝 API KEY: '%s' (get a free key at https://argovis-keygen.colorado.edu)"
                % api_key
            )
        else:
            summary.append("🗝 API KEY: '%s'" % api_key)
        return "\n".join(summary)

    def _add_history(self, this, txt):
        if "history" in this.attrs:
            this.attrs["history"] += "; %s" % txt
        else:
            this.attrs["history"] = txt
        return this

    @property
    def cachepath(self):
        """Return path to cache file for this request"""
        return [self.fs.cachepath(url) for url in self.uri]

    def cname(self):
        """Return a unique string defining the constraints"""
        return self._cname()

    def url_encode(self, urls):
        """Return safely encoded list of urls"""

        # return urls
        def safe_for_fsspec_cache(url):
            url = url.replace("[", "%5B")  # This is the one really necessary
            url = url.replace("]", "%5D")  # For consistency
            return url

        return [safe_for_fsspec_cache(url) for url in urls]

    def to_dataframe(self, errors: str = "ignore") -> pd.DataFrame:
        """Load Argo data and return a Pandas dataframe"""
        URI = self.uri  # Call it once

        # Download data:
        preprocess_opts = {"key_map": self.key_map}

        if not self.parallelize:
            if len(URI) == 1:
                data = self.fs.open_json(
                    URI[0],
                    errors=errors,
                    dwn_opts={'errors': errors},
                )
                df = pre_process(data, **preprocess_opts)

            else:
                df_list = self.fs.open_mfjson(
                    URI,
                    method=self.parallel_method,
                    preprocess=pre_process,
                    preprocess_opts=preprocess_opts,
                    open_json_opts={'errors': 'ignore',
                                    "download_url_opts": {"errors": "ignore"}
                                    },
                    progress=self.progress,
                    errors=errors,
                )
                df = pd.concat(df_list, ignore_index=True)

        else:
            df_list = self.fs.open_mfjson(
                URI,
                method=self.parallel_method,
                preprocess=pre_process,
                preprocess_opts=preprocess_opts,
                open_json_opts={'errors': 'ignore',
                                "download_url_opts": {"errors": "ignore"}
                                },
                progress=self.progress,
                errors=errors,
            )
            df = pd.concat(df_list, ignore_index=True)

        # Merge results (list of dataframe):
        if df.shape[0] == 0:
            raise DataNotFound("No data found for: %s" % self.cname())

        df.sort_values(by=["TIME", "PRES"], inplace=True)
        df["N_POINTS"] = np.arange(0, len(df["N_POINTS"]))
        df = df.set_index(["N_POINTS"])
        return df

    def to_xarray(self, errors: str = "ignore") -> xr.Dataset:
        """Download and return data as xarray Datasets"""
        ds = self.to_dataframe(errors=errors).to_xarray()
        # ds["TIME"] = pd.to_datetime(ds["TIME"], utc=True)
        ds = ds.sortby(
            ["TIME", "PRES"]
        )  # should already be sorted by date in descending order
        ds["N_POINTS"] = np.arange(
            0, len(ds["N_POINTS"])
        )  # Re-index to avoid duplicate values

        # Set coordinates:
        coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
        ds = ds.reset_coords()
        ds["N_POINTS"] = ds["N_POINTS"]
        # Convert all coordinate variable names to upper case
        for v in ds.data_vars:
            ds = ds.rename({v: v.upper()})
        ds = ds.set_coords(coords)

        # Add variable attributes and cast data types:
        ds = add_attributes(ds)
        ds = ds.argo.cast_types()

        # Remove argovis dataset attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.server
        try:
            ds.attrs["Fetched_by"] = getpass.getuser()
        except:  # noqa: E722
            ds.attrs["Fetched_by"] = "anonymous"
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = self.uri
        ds = ds[np.sort(ds.data_vars)]
        return ds

    def transform_data_mode(self, ds: xr.Dataset, **kwargs):
        # Argovis data are already curated !
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_data_mode(self, ds: xr.Dataset, **kwargs):
        # Argovis data are already curated !
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_qc(self, ds: xr.Dataset, **kwargs):
        # Argovis data are already curated !
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_researchmode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Filter dataset for research user mode

        This filter will select only QC=1 delayed mode data with pressure errors smaller than 20db

        Use this filter instead of transform_data_mode and filter_qc
        """
        ds = ds.argo.filter_researchmode()
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds


class Fetch_wmo(ArgovisDataFetcher):
    def init(self, WMO=[], CYC=None, **kwargs):
        """Create Argo data loader for WMOs and CYCs

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
            self.definition = "Argovis Argo data fetcher"
        if self.CYC is not None:
            self.definition = "%s for profiles" % self.definition
        else:
            self.definition = "%s for floats" % self.definition

        return self

    def get_url(self, wmo: int, cyc: int = None) -> str:
        """Return path toward the source file of a given wmo/cyc pair"""
        if cyc is None:
            return f"{self.server}/argo?platform={str(wmo)}&data=pressure,temperature,salinity"
        else:
            return f"{self.server}/argo?id={str(wmo)}_{str(cyc).zfill(3)}&data=pressure,temperature,salinity"

    @property
    def uri(self):
        """List of URLs to load for a request

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
                    this += [self.get_url(wmo, c) for c in cycs]
            return this

        urls = list_bunch(self.WMO, self.CYC)
        return self.url_encode(urls)


class Fetch_box(ArgovisDataFetcher):
    def init(self, box: list, **kwargs):
        """Create Argo data loader

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

    def get_url(self):
        """Return the URL used to download data"""
        shape = [[self.BOX[0], self.BOX[2]], [self.BOX[1], self.BOX[3]]]  # ll  # ur
        strShape = str(shape).replace(" ", "")
        url = self.server + "/argo?data=pressure,temperature,salinity&box=" + strShape
        url += "&startDate={}".format(
            pd.to_datetime(self.BOX[6]).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        url += "&endDate={}".format(
            pd.to_datetime(self.BOX[7]).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        url += "&presRange={},{}".format(self.BOX[4], self.BOX[5])
        return url

    @property
    def uri(self):
        """List of URLs to load for a request

        Returns
        -------
        list(str)
        """
        Lt = np.timedelta64(
            pd.to_datetime(self.BOX[7]) - pd.to_datetime(self.BOX[6]), "D"
        )
        MaxLenTime = 60
        MaxLen = np.timedelta64(MaxLenTime, "D")

        urls = []
        if not self.parallelize:
            # Check if the time range is not larger than allowed (MaxLenTime days):
            if Lt > MaxLen:
                self.Chunker = Chunker(
                    {"box": self.BOX},
                    chunks={"lon": 1, "lat": 1, "dpt": 1, "time": "auto"},
                    chunksize={"time": MaxLenTime},
                )
                boxes = self.Chunker.fit_transform()
                for box in boxes:
                    opts = {
                        "ds": self.dataset_id,
                        "fs": self.fs,
                        "server": self.server,
                    }
                    urls.append(Fetch_box(box=box, **opts).get_url())
            else:
                urls.append(self.get_url())
        else:
            if "time" not in self.chunks_maxsize:
                self.chunks_maxsize["time"] = MaxLenTime
            elif self.chunks_maxsize["time"] > MaxLenTime:
                warnings.warn(
                    (
                        "argovis only allows requests of %i days interval, "
                        "modify chunks_maxsize['time'] to %i" % (MaxLenTime, MaxLenTime)
                    )
                )
                self.chunks_maxsize["time"] = MaxLenTime

            # Ensure time chunks will never exceed what's allowed by argovis:
            if (
                Lt > MaxLen
                and "time" in self.chunks
                and self.chunks["time"] not in ["auto"]
            ):
                self.chunks["time"] = "auto"

            self.Chunker = Chunker(
                {"box": self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize
            )
            boxes = self.Chunker.fit_transform()
            for box in boxes:
                opts = {
                    "ds": self.dataset_id,
                    "fs": self.fs,
                    "server": self.server,
                }
                urls.append(
                    Fetch_box(
                        box=box,
                        **opts,
                    ).get_url()
                )

        return self.url_encode(urls)

"""
Fetcher to retrieve R/V CTD reference data for Argo DMQC from Ifremer erddap

This module is not available from the data fetcher facade because it does not provide data from
Argo floats but from R/V CTD.

This module is not covered by unit tests because it provide a preliminary support only (Ifremer erddap
data set not yet fully ready).

#todo Add unit tests when Ifremer erddap ready and new feature documented

"""
import xarray as xr
import logging

from erddapy.erddapy import ERDDAP  # noqa: F401
from erddapy.erddapy import _quote_string_constraints as quote_string_constraints  # noqa: F401
from erddapy.erddapy import parse_dates  # noqa: F401

from ..options import OPTIONS
from ..utils.chunking import Chunker
from ..utils.geo import conv_lon
from ..stores import httpstore_erddap_auth
from ..data_fetchers.erddap_data import ErddapArgoDataFetcher


log = logging.getLogger("argopy.erddap.refdata")

access_points = ["box"]
exit_formats = ["xarray"]
dataset_ids = ["ref-ctd"]  # First is default
api_server = OPTIONS["erddap"]  # API root url
api_server_check = (
    OPTIONS["erddap"] + "/info/ArgoFloats/index.json"
)  # URL to check if the API is alive


class ErddapREFDataFetcher(ErddapArgoDataFetcher):
    """Manage access to Argo CTD-reference data through Ifremer ERDDAP

    Examples
    --------
    >>> from argopy import CTDRefDataFetcher
    >>> with argopy.set_options(user="john_doe", password="***"):
    >>>      f = CTDRefDataFetcher(box=[15, 30, -70, -60, 0, 5000.0])
    >>>      ds = f.to_xarray()


    """

    # @doc_inherit
    def __init__(self, **kwargs):
        """Instantiate an authenticated ERDDAP Argo data fetcher

        Parameters
        ----------
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        api_timeout: int (optional)
            Erddap request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        kwargs["ds"] = "ref-ctd"
        super().__init__(**kwargs)
        kw = kwargs
        [
            kw.pop(p)
            for p in [
                "ds",
                "cache",
                "cachedir",
                "parallel",
                "parallel_method",
                "progress",
                "chunks",
                "chunks_maxsize",
                "api_timeout",
                "box",
            ]
            if p in kw
        ]
        login_page = "%s/login.html" % self.server.rstrip("/")
        self.fs = httpstore_erddap_auth(
            login=login_page, auto=False, **{**kw, **self.store_opts}
        )

    def __repr__(self):
        summary = [super().__repr__()]
        summary.append(
            "Performances: cache=%s, parallel=%s"
            % (str(self.fs.cache), str(self.parallel_method))
        )
        summary.append("User mode: %s" % "expert")
        summary.append("Dataset: %s" % self.dataset_id)
        return "\n".join(summary)

    def _add_attributes(self, this):  # noqa: C901
        """Add variables attributes not return by erddap requests

        This is hard coded, but should be retrieved from an API somewhere
        """
        this = super()._add_attributes(this)

        if "DIRECTION" in this.data_vars:
            this["DIRECTION"].attrs[
                "comment"
            ] = "Set to 'A' for all CTD stations by default"

        if "PLATFORM_NUMBER" in this.data_vars:
            this["PLATFORM_NUMBER"].attrs["long_name"] = "Fake unique identifier"
            this["PLATFORM_NUMBER"].attrs[
                "comment"
            ] = "This was inferred from EXPOCODE and is not a real WMO"

        if "CYCLE_NUMBER" in this.data_vars:
            this["CYCLE_NUMBER"].attrs["long_name"] = "Station number"
            this["CYCLE_NUMBER"].attrs[
                "comment"
            ] = "This was computed using unique TIME for each EXPOCODE"
            this["CYCLE_NUMBER"].attrs["convention"] = "-"

        return this

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(server=str(self.server), protocol="tabledap")
        self.erddap.response = "nc"
        self.erddap.dataset_id = "Argo-ref-ctd"
        return self

    @property
    def _minimal_vlist(self):
        """Return the minimal list of variables to retrieve measurements for"""
        # vlist = super()._minimal_vlist
        vlist = list()

        plist = ["latitude", "longitude", "time"]
        [vlist.append(p) for p in plist]
        plist = ["pres", "temp", "psal", "ptmp", "source", "qclevel"]
        [vlist.append(p) for p in plist]

        return vlist

    def to_xarray(self, errors: str = "ignore"):  # noqa: C901
        """Load CTD-Reference data and return a xarray.DataSet"""

        ds = super().to_xarray(errors=errors)

        ds = ds.rename({"SOURCE": "EXPOCODE"})
        ds["DIRECTION"] = xr.full_like(ds["EXPOCODE"], "A", dtype=str)
        g = []
        for iplatform, grp in enumerate(ds.groupby("EXPOCODE")):
            code, this_ds = grp
            for istation, sub_grp in enumerate(this_ds.groupby("TIME")):
                sub_grp[-1]["CYCLE_NUMBER"] = xr.full_like(
                    sub_grp[-1]["TIME"], istation, int
                )
                sub_grp[-1]["PLATFORM_NUMBER"] = xr.full_like(
                    sub_grp[-1]["TIME"], iplatform + 900000, int
                )
                g.append(sub_grp[-1])
        ds = xr.concat(
            g, dim="N_POINTS", data_vars="minimal", coords="minimal", compat="override"
        )

        ds.attrs["DATA_ID"] = "ARGO_Reference_CTD"
        ds.attrs["DOI"] = "-"

        # Cast data types and add variable attributes (not available in the csv download):
        ds = self._add_attributes(ds)
        ds = ds.argo.cast_types()

        return ds


class Fetch_box(ErddapREFDataFetcher):
    """Manage access to Argo CTD-reference data through Ifremer ERDDAP for: an ocean rectangle"""

    def init(self, box: list, **kw):
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
        self.definition = (
            "Ifremer erddap Argo CTD-REFERENCE data fetcher for a space/time region"
        )
        return self

    def define_constraints(self):
        """Define request constraints"""

        self.erddap.constraints = {"longitude>=": conv_lon(self.BOX[0], conv='360')}
        self.erddap.constraints.update({"longitude<=": conv_lon(self.BOX[1], conv='360')})
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
        """List of files to load for a request

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
                urls.append(
                    Fetch_box(
                        box=box, ds=self.dataset_id, fs=self.fs, server=self.server
                    ).get_url()
                )
            return urls

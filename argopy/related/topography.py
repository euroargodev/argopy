from typing import Union
from ..options import OPTIONS
from ..stores import httpstore
from ..utils.format import format_oneline


class TopoFetcher:
    """Fetch topographic data through an ERDDAP server for an ocean rectangle

    Example:
        >>> from argopy import TopoFetcher
        >>> box = [-75, -45, 20, 30]  # Lon_min, lon_max, lat_min, lat_max
        >>> ds = TopoFetcher(box).to_xarray()
        >>> ds = TopoFetcher(box, ds='gebco', stride=[10, 10], cache=True).to_xarray()

    """

    class ERDDAP:
        def __init__(self, server: str, protocol: str = "tabledap"):
            self.server = server
            self.protocol = protocol
            self.response = "nc"
            self.dataset_id = ""
            self.constraints = ""

    def __init__(
        self,
        box: list,
        ds: str = "gebco",
        cache: bool = False,
        cachedir: str = "",
        api_timeout: int = 0,
        stride: list = [1, 1],
        server: Union[str] = None,
        **kwargs,
    ):
        """Instantiate an ERDDAP topo data fetcher

        Parameters
        ----------
        ds: str (optional), default: 'gebco'
            Dataset to load:

            - 'gebco' will load the GEBCO_2020 Grid, a continuous terrain model for oceans and land at 15 arc-second intervals
        stride: list, default [1, 1]
            Strides along longitude and latitude. This allows to change the output resolution
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        api_timeout: int (optional)
            Erddap request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(
            cache=cache, cachedir=cachedir, timeout=timeout, size_policy="head"
        )
        self.definition = "Erddap topographic data fetcher"

        self.BOX = box
        self.stride = stride
        if ds == "gebco":
            self.definition = "NOAA erddap gebco data fetcher for a space region"
            self.server = (
                server
                if server is not None
                else "https://coastwatch.pfeg.noaa.gov/erddap"
            )
            self.server_name = "NOAA"
            self.dataset_id = "gebco"

        self._init_erddap()

    def _init_erddap(self):
        # Init erddap
        self.erddap = self.ERDDAP(server=self.server, protocol="griddap")
        self.erddap.response = "nc"

        if self.dataset_id == "gebco":
            self.erddap.dataset_id = "GEBCO_2020"
        else:
            raise ValueError(
                "Invalid database short name for %s erddap" % self.server_name
            )
        return self

    def _cname(self) -> str:
        """Fetcher one line string definition helper"""
        cname = "?"

        if hasattr(self, "BOX"):
            BOX = self.BOX
            cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % (
                BOX[0],
                BOX[1],
                BOX[2],
                BOX[3],
            )
        return cname

    def __repr__(self):
        summary = ["<topofetcher.erddap>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % self.server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def cname(self):
        """Return a unique string defining the constraints"""
        return self._cname()

    @property
    def cachepath(self):
        """Return path to cached file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(uri) for uri in self.uri]

    def define_constraints(self):
        """Define request constraints"""
        #        Eg: https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020.nc?elevation%5B(34):5:(42)%5D%5B(-21):7:(-12)%5D
        self.erddap.constraints = "%s(%0.2f):%i:(%0.2f)%s%s(%0.2f):%i:(%0.2f)%s" % (
            "%5B",
            self.BOX[2],
            self.stride[1],
            self.BOX[3],
            "%5D",
            "%5B",
            self.BOX[0],
            self.stride[0],
            self.BOX[1],
            "%5D",
        )
        return None

    #     @property
    #     def _minimal_vlist(self):
    #         """ Return the minimal list of variables to retrieve """
    #         vlist = list()
    #         vlist.append("latitude")
    #         vlist.append("longitude")
    #         vlist.append("elevation")
    #         return vlist

    def url_encode(self, url):
        """Return safely encoded list of urls

        This is necessary because fsspec cannot handle in cache paths/urls with a '[' character
        """

        # return urls
        def safe_for_fsspec_cache(url):
            url = url.replace("[", "%5B")  # This is the one really necessary
            url = url.replace("]", "%5D")  # For consistency
            return url

        return safe_for_fsspec_cache(url)

    def get_url(self):
        """Return the URL to download data requested

        Returns
        -------
        str
        """
        # First part of the URL:
        protocol = self.erddap.protocol
        dataset_id = self.erddap.dataset_id
        response = self.erddap.response
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"

        # Add variables to retrieve:
        variables = ["elevation"]
        variables = ",".join(variables)
        url += f"{variables}"

        # Add constraints:
        self.define_constraints()  # Define constraint to select this box of data (affect self.erddap.constraints)
        url += f"{self.erddap.constraints}"

        return self.url_encode(url)

    @property
    def uri(self):
        """List of files to load for a request

        Returns
        -------
        list(str)
        """
        return [self.get_url()]

    def to_xarray(self, errors: str = "ignore"):
        """Load Topographic data and return a xarray.DataSet"""

        # Download data
        if len(self.uri) == 1:
            ds = self.fs.open_dataset(self.uri[0])

        return ds

    def load(self, errors: str = "ignore"):
        """Load Topographic data and return a xarray.DataSet"""
        return self.to_xarray(errors=errors)

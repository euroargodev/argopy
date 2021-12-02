""" Argo file index store based on pyarrow """

import numpy as np
import pandas as pd
import logging
from abc import ABC

from argopy.options import OPTIONS
from argopy.stores import httpstore, memorystore
from argopy.errors import DataNotFound

try:
    import pyarrow.csv as csv
    import pyarrow as pa
except ModuleNotFoundError:
    pass


import gzip
import hashlib

log = logging.getLogger("argopy.stores.index.pa")


class ArgoIndexStoreProto(ABC):

    def _format(self, x, typ: str) -> str:
        """ string formatting helper """
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

    def cname(self) -> str:
        """ Fetcher one line string definition helper """
        cname = "?"

        if "BOX" in self.search_type:
            BOX = self.search_type['BOX']
            cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % (
                BOX[0],
                BOX[1],
                BOX[2],
                BOX[3],
            )
            if len(BOX) == 6:
                cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % (
                    BOX[0],
                    BOX[1],
                    BOX[2],
                    BOX[3],
                    self._format(BOX[4], "tim"),
                    self._format(BOX[5], "tim"),
                )

        elif "WMO" in self.search_type:
            WMO = self.search_type['WMO']
            if "CYC" in self.search_type:
                CYC = self.search_type['CYC']

            prtcyc = lambda CYC, wmo: "WMO%i_%s" % (  # noqa: E731
                wmo,
                "_".join(["CYC%i" % (cyc) for cyc in sorted(CYC)]),
            )

            if len(WMO) == 1:
                if "CYC" in self.search_type and isinstance(CYC, (np.ndarray)):
                    cname = "[%s]" % prtcyc(CYC, WMO[0])
                else:
                    cname = "[WMO%i]" % (WMO[0])
            else:
                cname = ";".join(["WMO%i" % wmo for wmo in sorted(WMO)])
                if "CYC" in self.search_type and isinstance(CYC, (np.ndarray)):
                    cname = ";".join([prtcyc(CYC, wmo) for wmo in WMO])
                cname = "[%s]" % cname

        cname = "index-%s" % cname
        return cname

    @property
    def sha(self) -> str:
        if self.cname() == "?":
            raise ValueError("Search not initialised")
        else:
            path = self.cname()
        return hashlib.sha256(path.encode()).hexdigest()


class indexstore(ArgoIndexStoreProto):

    def __init__(self,
                 host: str = "https://data-argo.ifremer.fr",
                 index_file: str = "ar_index_global_prof.txt",
                 cache: bool = False,
                 cachedir: str = "",
                 api_timeout: int = 0,
                 **kw):
        """ Create a file storage system for Argo index file requests

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (used value in global OPTIONS)
            index_file: str ("ar_index_global_prof.txt")
        """
        self.index_file = index_file
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        # self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')  # Only for https://data-argo.ifremer.fr
        self.fs = {}
        self.fs['index'] = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')  # Only for https://data-argo.ifremer.fr
        self.fs['search'] = memorystore(cache, cachedir)  # Manage the search results
        self.host = host

    def load(self, force=False):
        """ Load an Argo-index file content

        Try to load the gzipped file first, and if not found, fall back on the raw .txt file.

        Returns
        -------
        pyarrow.table
        """

        def read_csv(input_file):
            this_table = csv.read_csv(
                input_file,
                read_options=csv.ReadOptions(use_threads=True, skip_rows=8),
                convert_options=csv.ConvertOptions(
                    column_types={
                        'date': pa.timestamp('s', tz='utc'),
                        'date_update': pa.timestamp('s', tz='utc')
                    },
                    timestamp_parsers=['%Y%m%d%H%M%S']
                )
            )
            return this_table

        if not hasattr(self, 'index') or force:
            this_path = self.host + "/" + self.index_file
            if self.fs['index'].exists(this_path + '.gz'):
                with self.fs['index'].open(this_path + '.gz', "rb") as fg:
                    with gzip.open(fg) as f:
                        self.index = read_csv(f)
                        log.debug("Argo index file loaded with pyarrow read_csv from: %s" % f.name)
            else:
                with self.fs['index'].open(this_path, "rb") as f:
                    self.index = read_csv(f)
                    log.debug("Argo index file loaded with pyarrow read_csv from: %s" % f.name)

        return self

    def __repr__(self):
        summary = ["<argoindex>"]
        summary.append("Name: %s" % self.index_file)
        summary.append("FTP: %s" % self.host)
        if hasattr(self, 'index'):
            summary.append("Loaded: True (%i records)" % self.shape[0])
        else:
            summary.append("Loaded: False")
        if hasattr(self, 'search'):
            summary.append("Searched: True (%i records, %0.4f%%)" % (self.N_FILES, self.N_FILES * 100 / self.shape[0]))
        else:
            summary.append("Searched: False")
        return "\n".join(summary)

    @property
    def data(self):
        """ Return index as dataframe """
        return self.index.to_pandas()

    def to_dataframe(self):
        """ Return search results as dataframe

        If store is cached, caching is triggered here:
        """
        # df = self.search.to_pandas()
        if self.N_FILES == 0:
            raise DataNotFound("No Argo data in the index correspond to your search criteria."
                               "Search definition: %s" % self.cname())

        this_path = self.sha
        if self.fs['search'].exists(this_path):
            log.debug('Search results already in memory')
            with self.fs['search'].fs.open(this_path, "rb") as of:
                df = pd.read_pickle(of)
        else:
            log.debug('Search results conversion from scratch')
            df = self.search.to_pandas()  # Convert pyarrow table to pandas dataframe
            if self.cache:
                log.debug('Search results conversion saved in cache')
                with self.fs['search'].open(this_path, "wb") as of:
                    df.to_pickle(of) # Save in "memory"
                with self.fs['search'].fs.open(this_path, "rb") as of:
                    df = pd.read_pickle(of)  # Trigger save in cache file
        return df

    @property
    def full_uri(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.index['file']]

    @property
    def uri(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.search['file']]

    @property
    def shape(self):
        return self.index.shape

    @property
    def N_FILES(self):
        if hasattr(self, 'search'):
            return self.search.shape[0]
        else:
            return self.index.shape[0]

    def search_wmo(self, WMOs):
        self.load()
        self.search_type = {'WMO': WMOs}
        filt = []
        for wmo in WMOs:
            filt.append(pa.compute.match_substring_regex(self.index['file'], pattern="/%i/" % wmo))
        filt = np.logical_or.reduce(filt)
        self.search = self.index.filter(filt)
        log.debug("Argo index searched for WMOs=[%s]" % ";".join([str(wmo) for wmo in WMOs]))
        return self

    def search_wmo_cyc(self, WMOs, CYCs):
        self.load()
        self.search_type = {'WMO': WMOs, 'CYC': CYCs}
        filt = []
        for wmo in WMOs:
            for cyc in CYCs:
                if cyc < 1000:
                    pattern = "%i_%0.3d.nc" % (wmo, cyc)
                else:
                    pattern = "%i_%0.4d.nc" % (wmo, cyc)
                filt.append(pa.compute.match_substring_regex(self.index['file'], pattern=pattern))
        filt = np.logical_or.reduce(filt)
        self.search = self.index.filter(filt)
        log.debug("Argo index searched for WMOs=[%s] and CYCs=[%s]" % (
            ";".join([str(wmo) for wmo in WMOs]),
            ";".join([str(cyc) for cyc in CYCs])))
        return self

    def search_latlon(self, BOX):
        self.load()
        self.search_type = {'BOX': BOX}
        filt = []
        filt.append(pa.compute.greater_equal(self.index['longitude'], BOX[0]))
        filt.append(pa.compute.less_equal(self.index['longitude'], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index['latitude'], BOX[2]))
        filt.append(pa.compute.less_equal(self.index['latitude'], BOX[3]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        log.debug("Argo index searched for BOX=%s" % BOX)
        return self

    def search_tim(self, BOX):
        self.load()
        self.search_type = {'BOX': BOX}
        filt = []
        filt.append(pa.compute.greater_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                             pa.array([pd.to_datetime(BOX[4])], pa.timestamp('ms'))[0]))
        filt.append(pa.compute.less_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                          pa.array([pd.to_datetime(BOX[5])], pa.timestamp('ms'))[0]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        log.debug("Argo index searched for BOX=%s" % BOX)
        return self

    def search_latlontim(self, BOX):
        self.load()
        self.search_type = {'BOX': BOX}
        filt = []
        filt.append(pa.compute.greater_equal(self.index['longitude'], BOX[0]))
        filt.append(pa.compute.less_equal(self.index['longitude'], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index['latitude'], BOX[2]))
        filt.append(pa.compute.less_equal(self.index['latitude'], BOX[3]))
        filt.append(pa.compute.greater_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                             pa.array([pd.to_datetime(BOX[4])], pa.timestamp('ms'))[0]))
        filt.append(pa.compute.less_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                          pa.array([pd.to_datetime(BOX[5])], pa.timestamp('ms'))[0]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        log.debug("Argo index searched for BOX=%s" % BOX)
        return self


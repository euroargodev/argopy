""" Argo file index store based on pyarrow """

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

from argopy.options import OPTIONS
from argopy.stores import httpstore, memorystore
from argopy.errors import DataNotFound

try:
    import pyarrow.csv as csv
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pass


import gzip
import hashlib

log = logging.getLogger("argopy.stores.index.pa")


class ArgoIndexStoreProto(ABC):

    def __repr__(self):
        summary = ["<argoindex>"]
        summary.append("Index: %s" % self.index_file)
        summary.append("FTP: %s" % self.host)
        if hasattr(self, 'index'):
            summary.append("Loaded: True (%i records)" % self.shape[0])
        else:
            summary.append("Loaded: False")
        if hasattr(self, 'search'):
            match = 'matches' if self.N_FILES > 1 else 'match'
            summary.append("Searched: True (%i %s, %0.4f%%)" % (self.N_FILES, match, self.N_FILES * 100 / self.shape[0]))
        else:
            summary.append("Searched: False")
        return "\n".join(summary)

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
        """ Return the index constraints name as a string

         This method uses the BOX, WMO, CYC keys of the index instance ``search_type`` property
         """
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

        return cname

    @property
    def sha_df(self) -> str:
        """ Returns a unique SHA for a cname/dataframe """
        cname = "pd-%s" % self.cname()
        if cname == "?":
            raise ValueError("Search not initialised")
        else:
            path = cname
        return hashlib.sha256(path.encode()).hexdigest()

    @property
    def sha_pq(self) -> str:
        """ Returns a unique SHA for a cname/parquet """
        cname = "pq-%s" % self.cname()
        if cname == "?":
            raise ValueError("Search not initialised")
        else:
            path = cname
        return hashlib.sha256(path.encode()).hexdigest()

    @property
    def shape(self):
        """ Shape of in the index """
        # Should work for pyarrow.table and pandas.dataframe
        return self.index.shape

    @property
    def N_FILES(self):
        """ Number of files in search result, or index if search not triggered """
        # Should work for pyarrow.table and pandas.dataframe
        if hasattr(self, 'search'):
            return self.search.shape[0]
        else:
            return self.index.shape[0]

    @property
    @abstractmethod
    def data(self):
        """ Return index as dataframe """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri_full_index(self):
        """ List of URI from index

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri(self):
        """ List of URI from search results

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def load(self, force=False):
        """ Load an Argo-index file content

        Fill in self.index internal property
        If store is cached, caching is triggered here
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def run(self):
        """ Filter index with search criteria

        Fill in self.search internal property
        If store is cached, caching is triggered here
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def to_dataframe(self):
        """ Return search results as dataframe

        If store is cached, caching is triggered here
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_wmo(self):
        """ Return list of unique WMOs in search results

        Fall back on full index if search not found

        Returns
        -------
        list(int)
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def records_per_wmo(self):
        """ Return the number of records per unique WMOs in search results

        Fall back on full index if search not found

        Returns
        -------
        dict
            WMO are in keys, nb of records in values
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_wmo(self, WMOs):
        """ Search index for WMO

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        list(int)
            List of WMOs to search
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_wmo_cyc(self, WMOs, CYCs):
        """ Search index for WMO and CYC

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        list(int)
            List of WMOs to search
        list(int)
            List of CYCs to search
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_tim(self, BOX):
        """ Search index for time range

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        box : list()
            An index box to search Argo records for:

                - box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_latlon(self, BOX):
        """ Search index for rectangular latitude/longitude domain

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        box : list()
            An index box to search Argo records for:

                - box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_latlontim(self, BOX):
        """ Search index for rectangular latitude/longitude domain and time range

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        box : list()
            An index box to search Argo records for:

                - box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        raise NotImplementedError("Not implemented")


class indexstore(ArgoIndexStoreProto):
    """ Argo index store based on pyarrow table

    API Design:
    -----------
    The store is instantiated with the file name and access protocol:
    >>> idx = indexstore(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt", cache=True)

    Then, search can be carried with:
    >>> idx.search_wmo(1901393)
    >>> idx.search_wmo_cyc(1901393, [1,12])
    >>> idx.search_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
    >>> idx.search_latlon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])
    >>> idx.search_latlontim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

    Index and search properties:
    >>> idx.shape  # shape of the full index pyarrow table
    >>> idx.N_FILES  # Shortcut for length of 1st dimension of the search results pyarrow table (fall back on full index table if search not run)
    >>> idx.uri  # List of absolute path to files from the search results table column 'file'
    >>> idx.index  # Pyarrow table with full index
    >>> idx.search  # Pyarrow table with search results
    >>> idx.data  # Pandas dataframe with full index

    Internals:
    >>> idx.load()  # Load and read the full index file, using cache if necessary
    >>> idx.run()  # Run the search and save results in cache if necessary
    >>> idx.to_dataframe()  # Convert search results to pandas dataframe

    """

    def __init__(self,
                 host: str = "https://data-argo.ifremer.fr",
                 index_file: str = "ar_index_global_prof.txt",
                 cache: bool = False,
                 cachedir: str = "",
                 timeout: int = 0):
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
        timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
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
                        log.debug("Argo index file loaded with pyarrow read_csv from: '%s'" % f.name)
            else:
                with self.fs['index'].open(this_path, "rb") as f:
                    self.index = read_csv(f)
                    log.debug("Argo index file loaded with pyarrow read_csv from: %s" % f.name)

        return self

    def run(self):
        """ Filter index with search criteria """
        search_path = self.host + "/" + self.index_file + "/" + self.sha_pq
        if self.fs['search'].exists(search_path):
            log.debug('Search results already in memory as pyarrow table, loading...')
            with self.fs['search'].fs.open(search_path, "rb") as of:
                self.search = pa.parquet.read_table(of)
        else:
            log.debug('Compute search from scratch')
            self.search = self.index.filter(self.search_filter)
            log.debug('Found %i matches' % self.search.shape[0])
            if self.cache:
                with self.fs['search'].open(search_path, "wb") as of:
                    pa.parquet.write_table(self.search, of)
                with self.fs['search'].fs.open(search_path, "rb") as of:
                    self.search = pa.parquet.read_table(of)
                log.debug('Search results saved in cache as pyarrow table')
        return self

    def to_dataframe(self):
        """ Return search results as dataframe

        If store is cached, caching is triggered here
        """
        # df = self.search.to_pandas()
        if self.N_FILES == 0:
            raise DataNotFound("No Argo data in the index correspond to your search criteria."
                               "Search definition: %s" % self.cname())

        # this_path = self.sha_df
        this_path = self.host + "/" + self.index_file + "/" + self.sha_df
        if self.fs['search'].exists(this_path):
            log.debug('Search results already in memory as dataframe, loading...')
            with self.fs['search'].fs.open(this_path, "rb") as of:
                df = pd.read_pickle(of)
        else:
            log.debug('Search results conversion to dataframe from scratch')
            df = self.search.to_pandas()  # Convert pyarrow table to pandas dataframe
            if self.cache:
                with self.fs['search'].open(this_path, "wb") as of:
                    df.to_pickle(of) # Save in "memory"
                with self.fs['search'].fs.open(this_path, "rb") as of:
                    df = pd.read_pickle(of)  # Trigger save in cache file
                log.debug('Search results as dataframe saved in cache')
        return df

    @property
    def data(self):
        """ Return index as dataframe """
        return self.index.to_pandas()

    @property
    def uri_full_index(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.index['file']]

    @property
    def uri(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.search['file']]

    def read_wmo(self):
        """ Return list of unique WMOs in search results

        Fall back on full index if search not found

        Returns
        -------
        list(int)
        """
        if hasattr(self, 'search'):
            results = pa.compute.split_pattern(self.search['file'], pattern="/")
        else:
            results = pa.compute.split_pattern(self.index['file'], pattern="/")
        df = results.to_pandas()
        def fct(row):
            return row[1]
        wmo = df.map(fct)
        wmo = wmo.unique()
        wmo = [int(w) for w in wmo]
        return wmo

    def records_per_wmo(self):
        """ Return the number of records per unique WMOs in search results

            Fall back on full index if search not found
        """
        ulist = self.read_wmo()
        count = {}
        for wmo in ulist:
            if hasattr(self, 'search'):
                search_filter = pa.compute.match_substring_regex(self.search['file'], pattern="/%i/" % wmo)
                count[wmo] = self.search.filter(search_filter).shape[0]
            else:
                search_filter = pa.compute.match_substring_regex(self.index['file'], pattern="/%i/" % wmo)
                count[wmo] = self.index.filter(search_filter).shape[0]
        return count

    def search_wmo(self, WMOs):
        self.load()
        self.search_type = {'WMO': WMOs}
        filt = []
        for wmo in WMOs:
            filt.append(pa.compute.match_substring_regex(self.index['file'], pattern="/%i/" % wmo))
        self.search_filter = np.logical_or.reduce(filt)
        self.run()
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
        self.search_filter = np.logical_or.reduce(filt)
        self.run()
        log.debug("Argo index searched for WMOs=[%s] and CYCs=[%s]" % (
            ";".join([str(wmo) for wmo in WMOs]),
            ";".join([str(cyc) for cyc in CYCs])))
        return self

    def search_tim(self, BOX):
        self.load()
        self.search_type = {'BOX': BOX}
        filt = []
        filt.append(pa.compute.greater_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                             pa.array([pd.to_datetime(BOX[4])], pa.timestamp('ms'))[0]))
        filt.append(pa.compute.less_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                          pa.array([pd.to_datetime(BOX[5])], pa.timestamp('ms'))[0]))
        self.search_filter = np.logical_and.reduce(filt)
        self.run()
        log.debug("Argo index searched for BOX=%s" % BOX)
        return self

    def search_latlon(self, BOX):
        self.load()
        self.search_type = {'BOX': BOX}
        filt = []
        filt.append(pa.compute.greater_equal(self.index['longitude'], BOX[0]))
        filt.append(pa.compute.less_equal(self.index['longitude'], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index['latitude'], BOX[2]))
        filt.append(pa.compute.less_equal(self.index['latitude'], BOX[3]))
        self.search_filter = np.logical_and.reduce(filt)
        self.run()
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
        self.search_filter = np.logical_and.reduce(filt)
        self.run()
        log.debug("Argo index searched for BOX=%s" % BOX)
        return self


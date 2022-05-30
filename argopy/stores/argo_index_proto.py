"""
Argo file index store prototype

"""

import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from fsspec.core import split_protocol

from ..options import OPTIONS
from ..errors import FtpPathError, InvalidDataset, CacheFileNotFound
from ..utilities import Registry
from .filesystems import httpstore, memorystore, filestore, ftpstore

try:
    import pyarrow.csv as csv
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pass


log = logging.getLogger("argopy.stores.index")


class ArgoIndexStoreProto(ABC):
    """

        Examples
        --------

        An index store is instantiated with the access path (host) and the index file:

        >>> idx = indexstore()
        >>> idx = indexstore(host="ftp://ftp.ifremer.fr/ifremer/argo")
        >>> idx = indexstore(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt")
        >>> idx = indexstore(host="https://data-argo.ifremer.fr", index_file="ar_index_global_prof.txt", cache=True)

        Index methods and properties:

        >>> idx.load()
        >>> idx.load(nrows=12)  # Only load the first N rows of the index
        >>> idx.N_RECORDS  # Shortcut for length of 1st dimension of the index array
        >>> idx.index  # internal storage structure of the full index (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        >>> idx.shape  # shape of the full index array
        >>> idx.uri_full_index  # List of absolute path to files from the full index table column 'file'
        >>> idx.to_dataframe(index=True)  # Convert index to user-friendly :class:`pandas.DataFrame`
        >>> idx.to_dataframe(index=True, nrows=2)  # Only returns the first nrows of the index

        Search methods and properties:

        >>> idx.search_wmo(1901393)
        >>> idx.search_cyc(1)
        >>> idx.search_wmo_cyc(1901393, [1,12])
        >>> idx.search_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        >>> idx.search_lat_lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        >>> idx.search_lat_lon_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])  # Take an index BOX definition
        >>> idx.N_MATCH  # Shortcut for length of 1st dimension of the search results array
        >>> idx.search  # Internal table with search results
        >>> idx.uri  # List of absolute path to files from the search results table column 'file'
        >>> idx.run()  # Run the search and save results in cache if necessary
        >>> idx.to_dataframe()  # Convert search results to user-friendly :class:`pandas.DataFrame`
        >>> idx.to_dataframe(nrows=2)  # Only returns the first nrows of the search results


        Misc:

        >>> idx.cname
        >>> idx.read_wmo
        >>> idx.records_per_wmo

    """

    backend = "?"
    """Name of store backend"""

    search_type = {}
    """Dictionary with search meta-data"""

    ext = None
    """Storage file extension"""

    def __init__(
        self,
        host: str = "https://data-argo.ifremer.fr",
        index_file: str = "ar_index_global_prof.txt",
        cache: bool = False,
        cachedir: str = "",
        timeout: int = 0,
    ):
        """ Create an Argo index file store

            Parameters
            ----------
            host: str, default: ``https://data-argo.ifremer.fr``
                Host is a local or remote ftp/http path to a `dac` folder (GDAC structure compliant). This takes values
                like: ``ftp://ftp.ifremer.fr/ifremer/argo``, ``ftp://usgodae.org/pub/outgoing/argo`` or a local absolute path.
            index_file: str, default: ``ar_index_global_prof.txt``
                Name of the csv-like text file with the index
            cache : bool, default: False
                Use cache or not.
            cachedir : str, default: OPTIONS['cachedir'])
                Folder where to store cached files
        """
        self.host = host
        self.index_file = index_file
        self.cache = cache
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self.fs = {}
        if split_protocol(host)[0] is None:
            self.fs["src"] = filestore(cache=cache, cachedir=cachedir)
        elif "https" in split_protocol(host)[0]:
            # Only for https://data-argo.ifremer.fr (much faster than the ftp servers)
            self.fs["src"] = httpstore(
                cache=cache, cachedir=cachedir, timeout=timeout, size_policy="head"
            )
        elif "ftp" in split_protocol(host)[0]:
            if "ifremer" not in host and "usgodae" not in host:
                raise FtpPathError("""Unknown Argo ftp: %s. Raise on issue if you wish to add your own to the
                                   valid list of FTP servers:
                                   https://github.com/euroargodev/argopy/issues/new?title=New%%20FTP%%20server""" % host)
            self.fs["src"] = ftpstore(
                host=split_protocol(host)[-1].split("/")[0],  # host eg: ftp.ifremer.fr
                cache=cache,
                cachedir=cachedir,
                timeout=timeout,
                block_size=1000 * (2 ** 20),
            )
        else:
            raise FtpPathError(
                "Unknown protocol for an Argo index store: %s" % split_protocol(host)[0]
            )
        self.fs["client"] = memorystore(cache, cachedir)  # Manage search results
        self._memory_store_content = Registry(name='memory store') # Track files opened with this memory store, since it's a global store
        self.search_path_cache = Registry(name='cached search') # Track cached files related to search

        self.index_path = self.fs["src"].fs.sep.join([self.host, self.index_file])
        if not self.fs["src"].exists(self.index_path):
            raise FtpPathError("Index file does not exist: %s" % self.index_path)

    def __repr__(self):
        summary = ["<argoindex.%s>" % self.backend]
        summary.append("Host: %s" % self.host)
        summary.append("Index: %s" % self.index_file)
        if hasattr(self, "index"):
            summary.append("Loaded: True (%i records)" % self.N_RECORDS)
        else:
            summary.append("Loaded: False")
        if hasattr(self, "search"):
            match = "matches" if self.N_MATCH > 1 else "match"
            summary.append(
                "Searched: True (%i %s, %0.4f%%)"
                % (self.N_MATCH, match, self.N_MATCH * 100 / self.N_RECORDS)
            )
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

    @property
    def cname(self) -> str:
        """ Return the search constraint(s) as a pretty formatted string

            Return 'full' if a search was not yet performed on the indexstore instance

            This method uses the BOX, WMO, CYC keys of the index instance ``search_type`` property
         """
        cname = "full"

        if "BOX" in self.search_type:
            BOX = self.search_type["BOX"]
            cname = ("x=%0.2f/%0.2f;y=%0.2f/%0.2f") % (BOX[0], BOX[1], BOX[2], BOX[3],)
            if len(BOX) == 6:
                cname = ("x=%0.2f/%0.2f;y=%0.2f/%0.2f;t=%s/%s") % (
                    BOX[0],
                    BOX[1],
                    BOX[2],
                    BOX[3],
                    self._format(BOX[4], "tim"),
                    self._format(BOX[5], "tim"),
                )

        elif "WMO" in self.search_type:
            WMO = self.search_type["WMO"]
            if "CYC" in self.search_type:
                CYC = self.search_type["CYC"]

            prtcyc = lambda CYC, wmo: "WMO%i_%s" % (  # noqa: E731
                wmo,
                "_".join(["CYC%i" % (cyc) for cyc in sorted(CYC)]),
            )

            if len(WMO) == 1:
                if "CYC" in self.search_type:
                    cname = "%s" % prtcyc(CYC, WMO[0])
                else:
                    cname = "WMO%i" % (WMO[0])
            else:
                cname = ";".join(["WMO%i" % wmo for wmo in sorted(WMO)])
                if "CYC" in self.search_type:
                    cname = ";".join([prtcyc(CYC, wmo) for wmo in WMO])
                cname = "%s" % cname

        elif "CYC" in self.search_type and "WMO" not in self.search_type:
            CYC = self.search_type["CYC"]
            if len(CYC) == 1:
                cname = "CYC%i" % (CYC[0])
            else:
                cname = ";".join(["CYC%i" % cyc for cyc in sorted(CYC)])
            cname = "%s" % cname

        return cname

    def _sha_from(self, path):
        """ Internal post-processing for a sha

            Used by: sha_df, sha_pq, sha_h5
        """
        sha = path  # no encoding
        # sha = hashlib.sha256(path.encode()).hexdigest()  # Full encoding
        # log.debug("%s > %s" % (path, sha))
        return sha

    @property
    def sha_df(self) -> str:
        """ Returns a unique SHA for a cname/dataframe """
        cname = "pd-%s" % self.cname
        sha = self._sha_from(cname)
        return sha

    @property
    def sha_pq(self) -> str:
        """ Returns a unique SHA for a cname/parquet """
        cname = "pq-%s" % self.cname
        # if cname == "full":
        #     raise ValueError("Search not initialised")
        # else:
        #     path = cname
        sha = self._sha_from(cname)
        return sha

    @property
    def sha_h5(self) -> str:
        """ Returns a unique SHA for a cname/hdf5 """
        cname = "h5-%s" % self.cname
        # if cname == "full":
        #     raise ValueError("Search not initialised")
        # else:
        #     path = cname
        sha = self._sha_from(cname)
        return sha

    @property
    def shape(self):
        """ Shape of the index array """
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        return self.index.shape

    @property
    def N_FILES(self):
        """ Number of rows in search result or index if search not triggered """
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "search"):
            return self.search.shape[0]
        elif hasattr(self, "index"):
            return self.index.shape[0]
        else:
            raise InvalidDataset("You must, at least, load the index first !")

    @property
    def N_RECORDS(self):
        """ Number of rows in the full index """
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "index"):
            return self.index.shape[0]
        else:
            raise InvalidDataset("Load the index first !")

    @property
    def N_MATCH(self):
        """ Number of rows in search result """
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "search"):
            return self.search.shape[0]
        else:
            raise InvalidDataset("Initialised search first !")

    def _same_origin(self, path):
        """Compare origin of path with current memory fs"""
        return path in self._memory_store_content

    def _commit(self, path):
        self._memory_store_content.commit(path)

    def _write(self, fs, path, obj, fmt="pq"):
        """ Write internal array object to file store

            Parameters
            ----------
            fs: filestore
            obj: :class:`pyarrow.Table` or :class:`pandas.DataFrame`
            fmt: str
                File format to use. This is "pq" (default) or "pd"
        """
        this_path = path
        write_this = {
            'pq': lambda o, h: pa.parquet.write_table(o, h),
            'pd': lambda o, h: o.to_pickle(h)  # obj is a pandas dataframe
        }
        if fmt == 'parquet':
            fmt = 'pq'
        with fs.open(this_path, "wb") as handle:
            write_this[fmt](obj, handle)
            if fs.protocol == 'memory':
                self._commit(this_path)

        return self

    def _read(self, fs, path, fmt="pq"):
        """ Read internal array object from file store

            Parameters
            ----------
            fs: filestore
            path:
                Path to readable object
            fmt: str
                File format to use. This is "pq" (default) or "pd"

            Returns
            -------
            obj: :class:`pyarrow.Table` or :class:`pandas.DataFrame`
        """
        this_path = path
        read_this = {
            'pq': lambda h: pa.parquet.read_table(h),
            'pd': lambda h: pd.read_pickle(h)
        }
        if fmt == 'parquet':
            fmt = 'pq'
        with fs.open(this_path, "rb") as handle:
            obj = read_this[fmt](handle)
        return obj

    def clear_cache(self):
        """Clear cache registry and files associated with this store instance."""
        self.fs["src"].clear_cache()
        self.fs["client"].clear_cache()
        self._memory_store_content.clear()
        self.search_path_cache.clear()
        return self

    def cachepath(self, path):
        """ Return path to a cached file

        Parameters
        ----------
        path: str
            Path for which to return the cached file path for. You can use `index` or `search` as shortcuts
            to access path to the internal index or search tables.

        Returns
        -------
        list(str)
        """
        if path == 'index' and hasattr(self, 'index_path_cache'):
            path = [self.index_path_cache]
        elif path == 'search':
            if len(self.search_path_cache) > 0:
                path = self.search_path_cache.data
            else:
                path = [None]
            # elif not self.fs['client'].cache:
            #     raise
            # elif self.fs['client'].cache:
            #     raise
        elif not isinstance(path, list):
            path = [path]
        return [self.fs["client"].cachepath(p) for p in path]

    def to_dataframe(self, nrows=None, index=False):  # noqa: C901
        """ Return index or search results as :class:`pandas.DataFrame`

            If search not triggered, fall back on full index by default. Using index=True force to return the full index.

            Parameters
            ----------
            nrows: {int, None}, default: None
                Will return only the first `nrows` of search results. None returns all.
            index: bool, default: False
                Force to return the index, even if a search was performed with this store instance.

            Returns
            -------
            :class:`pandas.DataFrame`
        """
        def get_filename(s, index):
            if hasattr(self, "search") and not index:
                fname = s.search_path
            else:
                fname = s.index_path
            if nrows is not None:
                fname = fname + "/export" + "#%i.pd" % nrows
            else:
                fname = fname + "/export.pd"
            return fname

        df, src = self._to_dataframe(nrows=nrows, index=index)

        fname = get_filename(self, index)

        if self.cache and self.fs["client"].exists(fname):
            log.debug(
                "[%s] already processed as Dataframe, loading ... src='%s'"
                % (src, fname)
            )
            df = self._read(self.fs["client"].fs, fname, fmt="pd")
        else:
            log.debug("Converting [%s] to dataframe from scratch ..." % src)
            # Post-processing for user:
            from argopy.utilities import load_dict, mapp_dict

            if nrows is not None:
                df = df.loc[0 : nrows - 1].copy()

            if "index" in df:
                df.drop("index", axis=1, inplace=True)

            df.reset_index(drop=True, inplace=True)

            df["wmo"] = df["file"].apply(lambda x: int(x.split("/")[1]))
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
            df["date_update"] = pd.to_datetime(df["date_update"], format="%Y%m%d%H%M%S")

            # institution & profiler mapping for all users
            # todo: may be we need to separate this for standard and expert users
            institution_dictionnary = load_dict("institutions")
            df["tmp1"] = df["institution"].apply(
                lambda x: mapp_dict(institution_dictionnary, x)
            )
            df = df.rename(
                columns={"institution": "institution_code", "tmp1": "institution"}
            )

            profiler_dictionnary = load_dict("profilers")
            profiler_dictionnary["?"] = "?"

            def ev(x):
                try:
                    return int(x)
                except Exception:
                    return x

            df["profiler"] = df["profiler_type"].apply(
                lambda x: mapp_dict(profiler_dictionnary, ev(x))
            )
            df = df.rename(columns={"profiler_type": "profiler_code"})

            if self.cache:
                self._write(self.fs["client"], fname, df, fmt="pd")
                df = self._read(self.fs["client"].fs, fname, fmt="pd")
                if not index:
                    self.search_path_cache.commit(fname)  # Keep track of files related to search results
                log.debug("This dataframe saved in cache. dest='%s'" % fname)

        return df

    @property
    @abstractmethod
    def search_path(self):
        """ Path to search result uri

        Returns
        -------
        str
        """
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
    def load(self, nrows=None, force=False):
        """ Load an Argo-index file content

        Fill in self.index internal property
        If store is cached, caching is triggered here

        Try to load the gzipped file first, and if not found, fall back on the raw .txt file.


        Parameters
        ----------
        force: bool, default: False
            Force to refresh the index stored with this store instance
        nrows: {int, None}, default: None
            Maximum number of index rows to load


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
    def _to_dataframe(self) -> pd.DataFrame:
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
        """ Search index for floats defined by their WMO

        - Define search method
        - Trigger self.run() to set self.search internal property

        Parameters
        ----------
        list(int)
            List of WMOs to search

        Examples
        --------
        >>> idx.search_wmo(2901746)
        >>> idx.search_wmo([2901746, 4902252])
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_cyc(self, CYCs):
        """ Search index for cycle numbers

        Parameters
        ----------
        list(int)
            List of CYCs to search

        Examples
        --------
        >>> idx.search_cyc(1)
        >>> idx.search_cyc([1,2])
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_wmo_cyc(self, WMOs, CYCs):
        """ Search index for floats defined by their WMO and specific cycle numbers

        Parameters
        ----------
        list(int)
            List of WMOs to search
        list(int)
            List of CYCs to search

        Examples
        --------
        >>> idx.search_wmo_cyc(2901746, 12)
        >>> idx.search_wmo_cyc([2901746, 4902252], [1,2])
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_tim(self, BOX):
        """ Search index for a time range

        Parameters
        ----------
        box : list()
            An index box to search Argo records for.

        Warnings
        --------
        Only date bounds are considered from the index box.

        Examples
        --------
        >>> idx.search_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_lat_lon(self, BOX):
        """ Search index for a rectangular latitude/longitude domain

        Parameters
        ----------
        box : list()
            An index box to search Argo records for.

        Warnings
        --------
        Only lat/lon bounds are considered  from the index box.

        Examples
        --------
        >>> idx.search_lat_lon([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def search_lat_lon_tim(self, BOX):
        """ Search index for a rectangular latitude/longitude domain and time range

        Parameters
        ----------
        box : list()
            An index box to search Argo records for.

        Examples
        --------
        >>> idx.search_lat_lon_tim([-60, -55, 40., 45., '2007-08-01', '2007-09-01'])

        """
        raise NotImplementedError("Not implemented")


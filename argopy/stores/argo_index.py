import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import hashlib
import fsspec

from argopy.options import OPTIONS
from .fsspec_wrappers import filestore


class indexfilter_proto(ABC):
    """ Class prototype for an Argo index filter

    Such classes requires a ``run`` and ``uri`` methods.


    """
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        """ Take a _io.TextIOWrapper and return filtered data as string (csv likes)

        Parameters
        ----------
        index_file: _io.TextIOWrapper

        Returns
        -------
        csv rows matching the request, as a in-memory string. Or None.
        """
        pass

    @abstractmethod
    def uri(self):
        """ Return a name for one specific filter run """
        pass

    @property
    def sha(self):
        """ Unique filter hash string """
        return hashlib.sha256(self.uri().encode()).hexdigest()


class indexfilter_wmo(indexfilter_proto):
    """ Index filter based on WMO and/or CYCLE_NUMER

    This is intended to be used by instances of an indexstore

    Examples
    --------

    # Create filters:
    filt = index_filter_wmo(WMO=13857)
    filt = index_filter_wmo(WMO=13857, CYC=np.arange(1,10))
    filt = index_filter_wmo(WMO=[13857, 13858, 12], CYC=12)
    filt = index_filter_wmo(WMO=[13857, 13858, 12], CYC=[1, 12])
    filt = index_filter_wmo(CYC=250)
    filt = index_filter_wmo()

    # Filter name:
    print(filt.uri())

    # Direct usage:
        with open("/Volumes/Data/ARGO/ar_index_global_prof.txt", "r") as f:
            results = filt.run(f)

    # With the indexstore:
        indexstore(cache=1, index_file="/Volumes/Data/ARGO/ar_index_global_prof.txt").open_dataframe(filt)

    """

    def __init__(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo index filter for WMOs/CYCs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to search
            CYC : int, np.array(int), list(int)
                The cycle numbers to search for each WMO
        """
        if isinstance(WMO, int):
            WMO = [WMO]  # Make sure we deal with a list
        if isinstance(CYC, int):
            CYC = np.array((CYC,), dtype='int')  # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(CYC, dtype='int')  # Make sure we deal with an array of integers
        self.WMO = sorted(WMO)
        self.CYC = CYC

    def uri(self):
        """ Return a unique name for this filter instance """
        if len(self.WMO) > 1:
            listname = ["WMO%i" % i for i in sorted(self.WMO)]
            if isinstance(self.CYC, (np.ndarray)):
                [listname.append("CYC%0.4d" % i) for i in sorted(self.CYC)]
            listname = "_".join(listname)
        elif len(self.WMO) == 0:
            if isinstance(self.CYC, (np.ndarray)):
                listname = ["AllWMOs"]
                [listname.append("CYC%0.4d" % i) for i in sorted(self.CYC)]
            else:
                listname = ["FULL"]
            listname = "_".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
            if isinstance(self.CYC, (np.ndarray)):
                listname = [listname]
                [listname.append("CYC%0.4d" % i) for i in sorted(self.CYC)]
                listname = "_".join(listname)
        if len(listname) > 256:
            listname = hashlib.sha256(listname.encode()).hexdigest()
        return listname

    def run(self, index_file):
        """ Run search on an Argo index file

        Parameters
        ----------
        index_file: _io.TextIOWrapper

        Returns
        -------
        csv rows matching the request, as in-memory string. Or None.
        """

        # Define one-line search functions:
        def search_one_wmo(index, wmo):
            """ Search for a WMO in the argo index file

            Parameters
            ----------
            index_file: _io.TextIOWrapper
            wmo: int

            Returns
            -------
            csv chunk matching the request, as a string. Or None
            """
            index.seek(0)
            results = ""
            il_read, il_loaded, il_this = 0, 0, 0
            for line in index:
                il_this = il_loaded
                # if re.search("/%i/" % wmo, line.split(',')[0]):
                if "/%i/" % wmo in line:  # much faster than re
                    # Search for the wmo at the beginning of the file name under: /<dac>/<wmo>/profiles/
                    results += line
                    il_loaded += 1
                if il_this == il_loaded and il_this > 0:
                    break  # Since the index is sorted, once we found the float, we can stop reading the index !
                il_read += 1
            if il_loaded > 0:
                return results
            else:
                return None

        def search_any_wmo_cyc(index, cyc):
            """ Search for a WMO in the argo index file

            Parameters
            ----------
            index_file: _io.TextIOWrapper
            cyc: array of integers

            Returns
            -------
            csv chunk matching the request, as a string. Or None
            """

            def search_this(this_line):
                # return np.any([re.search("%0.3d.nc" % c, this_line.split(',')[0]) for c in cyc])
                return np.any(["%0.3d.nc" % c in this_line for c in cyc])
                if np.all(cyc >= 1000):
                    def search_this(this_line):
                        # return np.any([re.search("%0.4d.nc" % c, this_line.split(',')[0]) for c in cyc])
                        return np.any(["%0.4d.nc" % c in this_line for c in cyc])

            index.seek(0)
            results = ""
            il_read, il_loaded = 0, 0
            for line in index:
                if search_this(line):
                    results += line
                    il_loaded += 1
                il_read += 1
            if il_loaded > 0:
                return results
            else:
                return None

        def search_one_wmo_cyc(index, wmo, cyc):
            """ Search for a WMO and CYC in the argo index file

            Parameters
            ----------
            index: _io.TextIOWrapper
            wmo: int
            cyc: array of integers

            Returns
            -------
            csv chunk matching the request, as a string. Or None
            """
            index.seek(0)
            results = ""

            # Look for the float:
            il_read, il_loaded, il_this = 0, 0, 0
            for line in index:
                il_this = il_loaded
                # if re.search("/%i/" % wmo, line.split(',')[0]):
                if "/%i/" % wmo in line:  # much faster than re
                    results += line
                    il_loaded += 1
                if il_this == il_loaded and il_this > 0:
                    break  # Since the index is sorted, once we found the float, we can stop reading the index !
                il_read += 1

            # Then look for the profile:
            if results:
                def search_this(this_line):
                    # return np.any([re.search("%0.3d.nc" % c, this_line.split(',')[0]) for c in cyc])
                    return np.any(["%0.3d.nc" % c in this_line for c in cyc])
                if np.all(cyc >= 1000):
                    def search_this(this_line):
                        # return np.any([re.search("%0.4d.nc" % c, this_line.split(',')[0]) for c in cyc])
                        return np.any(["%0.4d.nc" % c in this_line for c in cyc])
                il_loaded, cyc_results = 0, ""
                for line in results.split():
                    if search_this(line):
                        il_loaded += 1
                        cyc_results += line + "\n"
            if il_loaded > 0:
                return cyc_results
            else:
                return None

        def full_load(index):
            """ Return the full argo index file (without header)

            Parameters
            ----------
            index: _io.TextIOWrapper

            Returns
            -------
            csv index, as a string
            """
            index.seek(0)
            for line in index:
                if line[0] != '#':
                    break
            return index.read()

        # Run the filter with the appropriate one-line search
        if len(self.WMO) > 1:
            if isinstance(self.CYC, (np.ndarray)):
                return "".join([r for r in [search_one_wmo_cyc(index_file, w, self.CYC) for w in self.WMO] if r])
            else:
                return "".join([r for r in [search_one_wmo(index_file, w) for w in self.WMO] if r])
        elif len(self.WMO) == 0:  # Search for cycle numbers only
            if isinstance(self.CYC, (np.ndarray)):
                return search_any_wmo_cyc(index_file, self.CYC)
            else:
                # No wmo, No cyc, return the full index:
                return full_load(index_file)
        else:
            if isinstance(self.CYC, (np.ndarray)):
                return search_one_wmo_cyc(index_file, self.WMO[0], self.CYC)
            else:
                return search_one_wmo(index_file, self.WMO[0])


class indexstore():
    """" Use to manage access to a local Argo index and searches """
    def __init__(self,
                 cache: bool = False,
                 cachedir: str = "",
                 index_file: str = "ar_index_global_prof.txt",
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
        self.fs = {}
        self.fs['index'] = filestore(cache, cachedir)
        if not cache:
            self.fs['search'] = fsspec.filesystem("memory")
        else:
            self.fs['search'] = fsspec.filesystem("filecache",
                                                  target_protocol='memory',
                                                  cache_storage=OPTIONS['cachedir'] if cachedir == '' else cachedir,
                                                  expiry_time=86400, cache_check=10)
            self.fs['search'].load_cache()

    def in_cache(self, fs, uri):
        """ Return true if uri is cached """
        if not uri.startswith(fs.target_protocol):
            store_path = fs.target_protocol + "://" + uri
        else:
            store_path = uri
        fs.load_cache()
        return store_path in fs.cached_files[-1]

    def in_memory(self, fs, uri):
        """ Return true if uri is in the memory store """
        return uri in fs.store

    def open_index(self):
        return self.fs['index'].open(self.index_file, "r")

    def res2dataframe(self, results):
        """ Convert a csv like string into a DataFrame

            If one columns has a missing value, the row is skipped
        """
        cols_name = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution', 'date_update']
        cols_type = {'file': np.str, 'date': np.datetime64, 'latitude': np.float32, 'longitude': np.float32,
                     'ocean': np.str, 'profiler_type': np.str, 'institution': np.str, 'date_update': np.datetime64}
        data = [x.split(',') for x in results.split('\n') if ",," not in x]
        return pd.DataFrame(data, columns=cols_name).astype(cols_type)[:-1]

    def open_dataframe(self, search_cls):
        """ Run a search on an Argo index file and return a Pandas dataframe with results

        Parameters
        ----------
        scls: Class instance of type index_filter_proto

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        uri = search_cls.uri()
        with self.open_index() as f:
            if self.cache and (self.in_cache(self.fs['search'], uri) or self.in_memory(self.fs['search'], uri)):
                # print('Search already in memory, loading:', uri)
                with self.fs['search'].open(uri, "r") as of:
                    df = self.res2dataframe(of.read())
            else:
                # print('Running search from scratch ...')
                # Run search:
                results = search_cls.run(f)
                # and save results for caching:
                if self.cache:
                    with self.fs['search'].open(uri, "w") as of:
                        of.write(results)  # This happens in memory
                df = self.res2dataframe(results)
        return df

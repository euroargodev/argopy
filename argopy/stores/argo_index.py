""" Argo index store """

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import hashlib

from ..errors import DataNotFound
from ..options import OPTIONS
from .filesystems import filestore, memorystore


def safe_rewind(this_index_obj):
    """ Rewind io.TextIOWrapper if seekable """
    if this_index_obj.seekable():
        this_index_obj.seek(0)
    # except io.UnsupportedOperation:
    #     # print(type(this_index_obj))
    #     pass
    return this_index_obj


class indexfilter_proto(ABC):
    """ Class prototype for an Argo index filter

    Such classes requires a ``run`` and ``uri`` methods.


    """

    @abstractmethod
    def run(self):
        """ Take a class:`io.TextIOWrapper` and return filtered data as string (csv likes)

        Parameters
        ----------
        index_file: class:`io.TextIOWrapper`

        Returns
        -------
        csv rows matching the request, as in-memory string. Or None.
        """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri(self):
        """ Return a name for one specific filter run """
        raise NotImplementedError("Not implemented")

    @property
    def sha(self):
        """ Unique filter hash string """
        return hashlib.sha256(self.uri.encode()).hexdigest()

    def search_null(self, index):
        """ Perform a null search, ie return the full argo index file

        Parameters
        ----------
        index: :class:`io.TextIOWrapper`

        Returns
        -------
        csv index, as a string
        """
        safe_rewind(index)

        for line in index:
            if line[0] != '#':
                break
        return index.read()


class indexfilter_wmo(indexfilter_proto):
    """ Index filter based on WMO and/or CYCLE_NUMER

    This is intended to be used by instances of an indexstore

    Examples
    --------

    # Create filters:
    filt = indexfilter_wmo(WMO=13857)
    filt = indexfilter_wmo(WMO=13857, CYC=np.arange(1,10))
    filt = indexfilter_wmo(WMO=[13857, 13858, 12], CYC=12)
    filt = indexfilter_wmo(WMO=[13857, 13858, 12], CYC=[1, 12])
    filt = indexfilter_wmo(CYC=250)
    filt = indexfilter_wmo()

    # Filter name:
    print(filt.uri)

    # Direct usage:
        with open("/Volumes/Data/ARGO/ar_index_global_prof.txt", "r") as f:
            results = filt.run(f)

    # With the indexstore:
        indexstore(cache=1, index_file="/Volumes/Data/ARGO/ar_index_global_prof.txt").read_csv(filt)

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

    @property
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

    def define_search_this(self, cyc):
        """ Return a search function for a given cycle number """
        if np.all(cyc >= 1000):
            def search_this(this_line):
                # return np.any([re.search("%0.4d.nc" % c, this_line.split(',')[0]) for c in cyc])
                return np.any(["%0.4d.nc" % c in this_line for c in cyc])
        else:
            def search_this(this_line):
                # return np.any([re.search("%0.3d.nc" % c, this_line.split(',')[0]) for c in cyc])
                return np.any(["%0.3d.nc" % c in this_line for c in cyc])
        return search_this

    def search_one_wmo(self, index, wmo):
        """ Search for a WMO in an argo index file

        Parameters
        ----------
        index_file: _io.TextIOWrapper
        wmo: int

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """
        safe_rewind(index)
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

    def search_any_wmo_cyc(self, index, cyc):
        """ Search for a WMO in an argo index file

        Parameters
        ----------
        index_file: _io.TextIOWrapper
        cyc: array of integers

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """
        search_this = self.define_search_this(cyc)
        safe_rewind(index)
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

    def search_one_wmo_cyc(self, index, wmo, cyc):
        """ Search for a WMO and CYC in an argo index file

        Parameters
        ----------
        index: _io.TextIOWrapper
        wmo: int
        cyc: array of integers

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """
        safe_rewind(index)
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
            search_this = self.define_search_this(cyc)
            il_loaded, cyc_results = 0, ""
            for line in results.split():
                if search_this(line):
                    il_loaded += 1
                    cyc_results += line + "\n"

        if il_loaded > 0:
            return cyc_results
        else:
            return None

    def run(self, index_file):
        """ Run search on an Argo index file

        Parameters
        ----------
        index_file: class:`io.TextIOWrapper`
            Argo index text stream

        Returns
        -------
        csv rows matching the request, as in-memory string. Or None.
        """

        # Run the filter with the appropriate one-line search
        if len(self.WMO) > 1:
            if isinstance(self.CYC, (np.ndarray)):
                return "".join([r for r in [self.search_one_wmo_cyc(index_file, w, self.CYC) for w in self.WMO] if r])
            else:
                return "".join([r for r in [self.search_one_wmo(index_file, w) for w in self.WMO] if r])
        elif len(self.WMO) == 0:  # Search for cycle numbers only
            if isinstance(self.CYC, (np.ndarray)):
                return self.search_any_wmo_cyc(index_file, self.CYC)
            else:
                # No wmo, No cyc, return the full index:
                return self.search_null(index_file)
        else:
            if isinstance(self.CYC, (np.ndarray)):
                return self.search_one_wmo_cyc(index_file, self.WMO[0], self.CYC)
            else:
                return self.search_one_wmo(index_file, self.WMO[0])


class indexfilter_box(indexfilter_proto):
    """ Index filter based on LATITUDE, LONGITUDE, DATE

    This is intended to be used by instances of an indexstore

    Examples
    --------

    # Create filters:
    filt = indexfilter_box(BOX=[-70, -65, 30., 35.])
    filt = indexfilter_box(BOX=[-70, -65, 30., 35., '2012-01-01', '2012-06-30'])

    # Filter name:
    print(filt.uri)

    # Direct usage:
        with open("/Volumes/Data/ARGO/ar_index_global_prof.txt", "r") as f:
            results = filt.run(f)

    # With the indexstore:
        indexstore(cache=1, index_file="/Volumes/Data/ARGO/ar_index_global_prof.txt").read_csv(filt)

    """

    def __init__(self, BOX: list = [], **kwargs):
        """ Create Argo index filter for LATITUDE, LONGITUDE, DATE

            Parameters
            ----------
            box : list(float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        # is_indexbox(BOX)
        self.BOX = BOX

    def _format(self, x, typ):
        """ string formatting helper """
        if typ == 'lon':
            if x < 0:
                x = 360. + x
            return ("%05d") % (x * 100.)
        if typ == 'lat':
            return ("%05d") % (x * 100.)
        if typ == 'prs':
            return ("%05d") % (np.abs(x) * 10.)
        if typ == 'tim':
            return pd.to_datetime(x).strftime('%Y%m%d')
        return str(x)

    @property
    def uri(self):
        """ Return a unique name for this filter instance """
        BOX = self.BOX
        if len(BOX) == 4:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % (BOX[0], BOX[1], BOX[2], BOX[3])
        else:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % (BOX[0], BOX[1], BOX[2], BOX[3], BOX[4], BOX[5])
        # if len(boxname) > 256:
        #     boxname = hashlib.sha256(boxname.encode()).hexdigest()
        return hashlib.sha256(boxname.encode()).hexdigest()
        # return boxname

    def search_latlon(self, index):
        """ Search

        Parameters
        ----------
        index: _io.TextIOWrapper

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """
        safe_rewind(index)
        results = ""
        iv_lat, iv_lon = 2, 3
        il_loaded = 0
        for ii in range(0, 9):
            index.readline()
        for line in index:
            this_line = line.split(",")
            if this_line[iv_lon] != "" and this_line[iv_lat] != "":
                x = float(this_line[iv_lon])
                y = float(this_line[iv_lat])
                if x >= self.BOX[0] and x <= self.BOX[1] and y >= self.BOX[2] and y <= self.BOX[3]:
                    results += line
                    il_loaded += 1
        if il_loaded > 0:
            return results
        else:
            return None

    def search_tim(self, index):
        """ Search

        Parameters
        ----------
        index: str csv like

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """
        results = ""
        iv_tim = 1
        il_loaded = 0
        for line in index.split():
            this_line = line.split(",")
            if this_line[iv_tim] != "":
                t = pd.to_datetime(str(this_line[iv_tim]))
                if t >= pd.to_datetime(self.BOX[4]) and t <= pd.to_datetime(self.BOX[5]):
                    results += line + "\n"
                    il_loaded += 1
        if il_loaded > 0:
            return results
        else:
            return None

    def search_latlontim(self, index):
        """ Search

        Parameters
        ----------
        index: _io.TextIOWrapper

        Returns
        -------
        csv chunk matching the request, as a string. Or None
        """

        # First search in space:
        results = self.search_latlon(index)
        # Then refine in time:
        if results:
            results = self.search_tim(results)
        return results

    def run(self, index_file):
        """ Run search on an Argo index file

        Parameters
        ----------
        index_file: class:`io.TextIOWrapper`
            Argo index text stream

        Returns
        -------
        csv rows matching the request, as in-memory string. Or None.
        """

        # Run the filter:
        if len(self.BOX) == 4:
            return self.search_latlon(index_file)
        else:
            return self.search_latlontim(index_file)


class indexstore():
    """Legacy Argo index store.

    Examples
    --------
    BOX = [-60, -55, 40., 45., '2007-08-01', '2007-09-01']
    filt = indexfilter_box(BOX)
    df = indexstore(cache=True).read_csv(filt)
    """

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
        self.fs['index'] = filestore(cache, cachedir)  # Manage the full index
        self.fs['search'] = memorystore(cache, cachedir)  # Manage the search results

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        return self.fs['search'].cachepath(uri, errors)

    def clear_cache(self):
        self.fs['index'].clear_cache()
        self.fs['search'].clear_cache()

    # def in_cache(self, fs, uri):
    #     """ Return True if uri is cached """
    #     if not uri.startswith(fs.target_protocol):
    #         store_path = fs.target_protocol + "://" + uri
    #     else:
    #         store_path = uri
    #     fs.load_cache()
    #     return store_path in fs.cached_files[-1]

    # def in_memory(self, fs, uri):
    #     """ Return True if uri is in memory """
    #     return fs.exists(uri)

    # def open_index(self):
    #     return self.fs['index'].open(self.index_file, "r")

    def res2dataframe(self, results):
        """ Convert a csv like string into a DataFrame

            If one columns has a missing value, the row is skipped

        Parameters
        ----------
        str

        Returns
        -------
        :class:`pandas.Dataframe`
        """
        cols_name = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution', 'date_update']
        cols_type = {'file': np.str_, 'date': np.datetime64, 'latitude': np.float32, 'longitude': np.float32,
                     'ocean': np.str_, 'profiler_type': np.str_, 'institution': np.str_, 'date_update': np.datetime64}
        data = [x.split(',') for x in results.split('\n') if ",," not in x]
        return pd.DataFrame(data, columns=cols_name).astype(cols_type)[:-1]

    def read_csv(self, search):
        """ Run a search on an csv Argo index file and return a Pandas DataFrame with results

        Parameters
        ----------
        search: :class:`indexfilter_wmo` or :class:`indexfilter_box`
            Class instance inheriting from :class:`indexfilter_proto`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        if self.fs['search'].exists(search.uri):
            # print('\nSearch already in memory, loading:', search.uri)
            results = ""
            with self.fs['search'].fs.open(search.uri, "r") as of:
                results += of.readline()
        else:
            # print('\nRunning search from scratch ...')
            with self.fs['index'].open(self.index_file, "r") as f:
                # Run search:
                results = search.run(f)
                if not results:
                    raise DataNotFound("No data found in index: %s" % search.uri)
                # and save results for caching:
                if self.cache:
                    with self.fs['search'].open(search.uri, "w") as of:
                        of.write(results)  # Save in "memory"
                    results = ""
                    with self.fs['search'].fs.open(search.uri, "r") as of:
                        results += of.readline()  # Trigger save in cache file
        return self.res2dataframe(results)

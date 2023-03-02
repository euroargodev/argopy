"""
Argo file index store

Implementation based on pyarrow
"""

import numpy as np
import pandas as pd
import logging
import io
import gzip
from packaging import version

from ..errors import DataNotFound
from ..utilities import check_index_cols, is_indexbox, check_wmo, check_cyc, doc_inherit
from .argo_index_proto import ArgoIndexStoreProto
try:
    import pyarrow.csv as csv
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except ModuleNotFoundError:
    pass


log = logging.getLogger("argopy.stores.index.pa")


class indexstore_pyarrow(ArgoIndexStoreProto):
    """Argo GDAC index store using :class:`pyarrow.Table` as internal storage format.

    With this store, index and search results are saved as pyarrow/parquet files in cache

    """
    __doc__ += ArgoIndexStoreProto.__doc__

    backend = "pyarrow"

    ext = "pq"
    """Storage file extension"""

    @doc_inherit
    def load(self, nrows=None, force=False):
        """ Load an Argo-index file content

        Returns
        -------
        :class:`pyarrow.Table`
        """

        def read_csv(input_file, nrows=None):
            # pyarrow doesn't have a concept of 'nrows' but it's really important
            # for partial downloading of the giant prof index
            # This is totaly copied from: https://github.com/ArgoCanada/argopandas/blob/master/argopandas/global_index.py#L20
            if nrows is not None:
                buf = io.BytesIO()
                n = 0
                for line in input_file:
                    n += 1
                    buf.write(line)
                    if n >= (nrows + 8 + 1):
                        break

                buf.seek(0)
                return read_csv(buf, nrows=None)

            # log.debug("Index source file: %s (%s bytes)" % (type(input_file), sys.getsizeof(input_file)))
            # Possible input_file type:
            # _io.BufferedReader
            # _io.BytesIO
            # gzip.GzipFile
            this_table = csv.read_csv(
                input_file,
                read_options=csv.ReadOptions(use_threads=True, skip_rows=8),
                convert_options=csv.ConvertOptions(
                    column_types={
                        "date": pa.timestamp("s"),  # , tz="utc"
                        "date_update": pa.timestamp("s"),
                    },
                    timestamp_parsers=["%Y%m%d%H%M%S"],
                ),
            )
            # Using tz="utc" was raising this error:
            # pyarrow.lib.ArrowInvalid: In CSV column  # 7: CSV conversion error to timestamp[s, tz=utc]: expected a
            # zone offset in '20181011180520'. If these timestamps are in local time, parse them as timestamps without
            # timezone, then call assume_timezone. If using strptime, ensure '%z' is in the format string.
            # So I removed the option in c0a15ec68013c78d83f2689a8f9c062fdfa160ab
            return this_table

        def csv2index(obj, origin):
            index_file = origin.split(self.fs['src'].fs.sep)[-1]
            index = read_csv(obj, nrows=nrows)
            check_index_cols(
                index.column_names,
                convention=index_file.split(".")[0],
            )
            log.debug("Argo index file loaded with pyarrow read_csv. src='%s'" % origin)
            return index

        if not hasattr(self, "index") or force:
            this_path = self.index_path
            if nrows is not None:
                this_path = this_path + "/local" + "#%i.%s" % (nrows, self.ext)
            else:
                this_path = this_path + "/local.%s" % self.ext

            if self.cache and self.fs["client"].exists(this_path): # and self._same_origin(this_path):
                log.debug(
                    "Index already in memory as pyarrow table, loading... src='%s'"
                    % (this_path)
                )
                self.index = self._read(self.fs["client"].fs, this_path, fmt=self.ext)
                self.index_path_cache = this_path
            else:
                log.debug("Load index from scratch (nrows=%s) ..." % nrows)
                if self.fs["src"].exists(self.index_path + ".gz"):
                    with self.fs["src"].open(self.index_path + ".gz", "rb") as fg:
                        with gzip.open(fg) as f:
                            self.index = csv2index(f, self.index_path + ".gz")
                else:
                    with self.fs["src"].open(self.index_path, "rb") as f:
                        self.index = csv2index(f, self.index_path)

                if self.cache and self.index.shape[0] > 0:
                    self._write(self.fs["client"], this_path, self.index, fmt=self.ext)
                    self.index = self._read(self.fs["client"].fs, this_path)
                    self.index_path_cache = this_path
                    log.debug(
                        "Index saved in cache as pyarrow table. dest='%s'"
                        % this_path
                    )

        if self.N_RECORDS == 0:
            raise DataNotFound("No data found in the index")
        elif nrows is not None and self.N_RECORDS != nrows:
            self.index = self.index[0: nrows - 1]

        return self

    def run(self, nrows=None):
        """ Filter index with search criteria """
        this_path = self.search_path
        if nrows is not None:
            this_path = this_path + "/local" + "#%i.%s" % (nrows, self.ext)
        else:
            this_path = this_path + "/local.%s" % self.ext

        if self.cache and self.fs["client"].exists(this_path): # and self._same_origin(this_path):
            log.debug(
                "Search results already in memory as pyarrow table, loading... src='%s'"
                % (this_path)
            )
            self.search = self._read(self.fs["client"].fs, this_path, fmt=self.ext)
            self.search_path_cache.commit(this_path)
        else:
            log.debug("Compute search from scratch (nrows=%s) ..." % nrows)
            this_filter = np.nonzero(self.search_filter)[0]
            n_match = this_filter.shape[0]
            if nrows is not None and n_match > 0:
                self.search = self.index.take(
                    this_filter.take(range(np.min([nrows, n_match])))
                )
            else:
                self.search = self.index.filter(self.search_filter)

            log.debug("Found %i/%i matches" % (self.search.shape[0], self.index.shape[0]))
            if self.cache and self.search.shape[0] > 0:
                self._write(self.fs["client"], this_path, self.search, fmt=self.ext)
                self.search = self._read(self.fs["client"].fs, this_path)
                self.search_path_cache.commit(this_path)
                log.debug(
                    "Search results saved in cache as pyarrow table. dest='%s'"
                    % this_path
                )
        return self

    def _to_dataframe(self, nrows=None, index=False):  # noqa: C901
        """ Return search results as dataframe

            If search not triggered, fall back on full index by default. Using index=True force to return the full index.

            This is where we can process the internal dataframe structure for the end user.
            If this processing is long, we can implement caching here.
        """
        if hasattr(self, "search") and not index:
            if self.N_MATCH == 0:
                raise DataNotFound(
                    "No data found in the index corresponding to your search criteria."
                    " Search definition: %s" % self.cname
                )
            else:
                src = "search results"
                df = self.search.to_pandas()
        else:
            src = "full index"
            if not hasattr(self, "index"):
                self.load(nrows=nrows)
            df = self.index.to_pandas()

        return df, src

    @property
    def search_path(self):
        """ Path to search result uri"""
        # return self.host + "/" + self.index_file + "." + self.sha_pq
        return self.fs["client"].fs.sep.join([self.host, "%s.%s" % (self.index_file, self.sha_pq)])

    @property
    def uri_full_index(self):
        # return ["/".join([self.host, "dac", f.as_py()]) for f in self.index["file"]]
        sep = self.fs["src"].fs.sep
        return [sep.join([self.host, "dac", f.as_py().replace('/', sep)]) for f in self.index["file"]]

    @property
    def uri(self):
        # return ["/".join([self.host, "dac", f.as_py()]) for f in self.search["file"]]
        #todo Should also modify separator from "f.as_py()" because it's "/" on the index file,
        # but should be turned to "\" for local file index on Windows. Remains "/" in all others (linux, mac, ftp. http)
        sep = self.fs["src"].fs.sep
        # log.warning("[sys sep=%s] vs [fs/src sep=%s]" % (os.path.sep, self.fs["src"].fs.sep))
        return [sep.join([self.host, "dac", f.as_py().replace('/', sep)]) for f in self.search["file"]]

    def read_wmo(self, index=False):
        """ Return list of unique WMOs in search results

        Fall back on full index if search not found

        Returns
        -------
        list(int)
        """
        if hasattr(self, "search") and not index:
            results = pa.compute.split_pattern(self.search["file"], pattern="/")
        else:
            results = pa.compute.split_pattern(self.index["file"], pattern="/")
        df = results.to_pandas()

        def fct(row):
            return row[1]

        wmo = df.map(fct)
        wmo = wmo.unique()
        wmo = [int(w) for w in wmo]
        return wmo

    def records_per_wmo(self, index=False):
        """ Return the number of records per unique WMOs in search results

            Fall back on full index if search not found
        """
        ulist = self.read_wmo()
        count = {}
        for wmo in ulist:
            if hasattr(self, "search") and not index:
                search_filter = pa.compute.match_substring_regex(
                    self.search["file"], pattern="/%i/" % wmo
                )
                count[wmo] = self.search.filter(search_filter).shape[0]
            else:
                search_filter = pa.compute.match_substring_regex(
                    self.index["file"], pattern="/%i/" % wmo
                )
                count[wmo] = self.index.filter(search_filter).shape[0]
        return count

    def _reduce_a_filter_list(self, filters, op='or'):
        if version.parse(pa.__version__) < version.parse("7.0"):
            filters = [i.to_pylist() for i in filters]
        if op == 'or':
            return np.logical_or.reduce(filters)
        elif op == 'and':
            return np.logical_and.reduce(filters)

    def search_wmo(self, WMOs, nrows=None):
        WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
        log.debug(
            "Argo index searching for WMOs=[%s] ..."
            % ";".join([str(wmo) for wmo in WMOs])
        )
        self.load()
        self.search_type = {"WMO": WMOs}
        filt = []
        for wmo in WMOs:
            filt.append(
                pa.compute.match_substring_regex(
                    self.index["file"], pattern="/%i/" % wmo
                )
            )
        self.search_filter = self._reduce_a_filter_list(filt)
        self.run(nrows=nrows)
        return self

    def search_cyc(self, CYCs, nrows=None):
        CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
        log.debug(
            "Argo index searching for CYCs=[%s] ..."
            % (";".join([str(cyc) for cyc in CYCs]))
        )
        self.load()
        self.search_type = {"CYC": CYCs}
        filt = []
        for cyc in CYCs:
            if cyc < 1000:
                pattern = "_%0.3d.nc" % (cyc)
            else:
                pattern = "_%0.4d.nc" % (cyc)
            filt.append(
                pa.compute.match_substring_regex(self.index["file"], pattern=pattern)
            )
        self.search_filter = self._reduce_a_filter_list(filt)
        self.run(nrows=nrows)
        return self

    def search_wmo_cyc(self, WMOs, CYCs, nrows=None):
        WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
        CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
        log.debug(
            "Argo index searching for WMOs=[%s] and CYCs=[%s] ..."
            % (
                ";".join([str(wmo) for wmo in WMOs]),
                ";".join([str(cyc) for cyc in CYCs]),
            )
        )
        self.load()
        self.search_type = {"WMO": WMOs, "CYC": CYCs}
        filt = []
        for wmo in WMOs:
            for cyc in CYCs:
                if cyc < 1000:
                    pattern = "%i_%0.3d.nc" % (wmo, cyc)
                else:
                    pattern = "%i_%0.4d.nc" % (wmo, cyc)
                filt.append(
                    pa.compute.match_substring_regex(
                        self.index["file"], pattern=pattern
                    )
                )
        self.search_filter = self._reduce_a_filter_list(filt)
        self.run(nrows=nrows)
        return self

    def search_tim(self, BOX, nrows=None):
        is_indexbox(BOX)
        log.debug("Argo index searching for time in BOX=%s ..." % BOX)
        self.load()
        self.search_type = {"BOX": BOX}
        filt = []
        filt.append(
            pa.compute.greater_equal(
                pa.compute.cast(self.index["date"], pa.timestamp("ms")),
                pa.array([pd.to_datetime(BOX[4])], pa.timestamp("ms"))[0],
            )
        )
        filt.append(
            pa.compute.less_equal(
                pa.compute.cast(self.index["date"], pa.timestamp("ms")),
                pa.array([pd.to_datetime(BOX[5])], pa.timestamp("ms"))[0],
            )
        )
        self.search_filter = self._reduce_a_filter_list(filt, op='and')
        self.run(nrows=nrows)
        return self

    def search_lat_lon(self, BOX, nrows=None):
        is_indexbox(BOX)
        log.debug("Argo index searching for lat/lon in BOX=%s ..." % BOX)
        self.load()
        self.search_type = {"BOX": BOX}
        filt = []
        filt.append(pa.compute.greater_equal(self.index["longitude"], BOX[0]))
        filt.append(pa.compute.less_equal(self.index["longitude"], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index["latitude"], BOX[2]))
        filt.append(pa.compute.less_equal(self.index["latitude"], BOX[3]))
        self.search_filter = self._reduce_a_filter_list(filt, op='and')
        self.run(nrows=nrows)
        return self

    def search_lat_lon_tim(self, BOX, nrows=None):
        is_indexbox(BOX)
        log.debug("Argo index searching for lat/lon/time in BOX=%s ..." % BOX)
        self.load()
        self.search_type = {"BOX": BOX}
        filt = []
        filt.append(pa.compute.greater_equal(self.index["longitude"], BOX[0]))
        filt.append(pa.compute.less_equal(self.index["longitude"], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index["latitude"], BOX[2]))
        filt.append(pa.compute.less_equal(self.index["latitude"], BOX[3]))
        filt.append(
            pa.compute.greater_equal(
                pa.compute.cast(self.index["date"], pa.timestamp("ms")),
                pa.array([pd.to_datetime(BOX[4])], pa.timestamp("ms"))[0],
            )
        )
        filt.append(
            pa.compute.less_equal(
                pa.compute.cast(self.index["date"], pa.timestamp("ms")),
                pa.array([pd.to_datetime(BOX[5])], pa.timestamp("ms"))[0],
            )
        )
        self.search_filter = self._reduce_a_filter_list(filt, op='and')
        self.run(nrows=nrows)
        return self

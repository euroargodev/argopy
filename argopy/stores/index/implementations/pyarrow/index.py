import numpy as np
import pandas as pd
import logging
import io
import gzip
from packaging import version
from pathlib import Path
from typing import List

try:
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401
    import pyarrow.compute as pc  # noqa: F401
except ModuleNotFoundError:
    pass

from .....options import OPTIONS
from .....errors import DataNotFound, InvalidDatasetStructure
from .....utils import check_index_cols, conv_lon
from ...spec import ArgoIndexStoreProto


log = logging.getLogger("argopy.stores.index.pa")


class indexstore(ArgoIndexStoreProto):
    """Argo GDAC index store using :class:`pyarrow.Table` as internal storage format.

    With this store, index and search results are saved as pyarrow/parquet files in cache

    """

    # __doc__ += ArgoIndexStoreProto.__doc__

    backend = "pyarrow"

    ext = "pq"
    """Storage file extension"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, nrows=None, force=False):  # noqa: C901
        """Load an Argo-index file content

        Returns
        -------
        self
        """

        def read_csv(input_file, nrows=None):
            # pyarrow doesn't have a concept of 'nrows' but it's really important
            # for partial downloading of the giant prof index
            # This is totally copied from: https://github.com/ArgoCanada/argopandas/blob/master/argopandas/global_index.py#L20
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
                read_options=csv.ReadOptions(
                    use_threads=True, skip_rows=self.skip_rows
                ),
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

        def csv2index(obj):
            index = read_csv(obj, nrows=nrows)
            # log.debug(index.column_names)
            check_index_cols(
                index.column_names,
                convention=self.convention,
            )
            if "longitude" in self.convention_columns:
                index = index.append_column(
                    "longitude_360",
                    pa.array(conv_lon(index["longitude"].to_numpy(), "360")),
                )
            return index

        def index2cache_path(path, nrows=None):
            if nrows is not None:
                cache_path = path + "/local" + "#%i.%s" % (nrows, self.ext)
            else:
                cache_path = path + "/local.%s" % self.ext
            return cache_path

        def download(nrows=None):
            log.debug("Load Argo index (nrows=%s) ..." % nrows)
            if Path(self.index_path).suffix == ".gz":
                with self.fs["src"].open(self.index_path, "rb") as fg:
                    with gzip.open(fg) as f:
                        self.index = csv2index(f)
            else:
                with self.fs["src"].open(self.index_path, "rb") as f:
                    self.index = csv2index(f)
            log.debug(
                "Argo index file loaded with Pyarrow csv.read_csv from '%s'"
                % self.index_path
            )
            self._nrows_index = nrows

        def save2cache(path_in_cache):
            self._write(self.fs["client"], path_in_cache, self.index, fmt=self.ext)
            self.index = self._read(self.fs["client"].fs, path_in_cache)
            self.index_path_cache = path_in_cache
            log.debug(
                "Argo index saved in cache as a Pyarrow table at '%s'" % path_in_cache
            )

        def loadfromcache(path_in_cache):
            log.debug(
                "Argo index already in cache as a Pyarrow table, loading from '%s'"
                % path_in_cache
            )
            self.index = self._read(self.fs["client"].fs, path_in_cache, fmt=self.ext)
            self.index_path_cache = path_in_cache

        index_path_cache = index2cache_path(self.index_path, nrows=nrows)

        if hasattr(self, "_nrows_index") and self._nrows_index != nrows:
            force = True

        if force:
            download(nrows=nrows)
            if self.cache:
                save2cache(index_path_cache)

        else:
            if not hasattr(self, "index") or (
                hasattr(self, "index") and getattr(self, "index") is None
            ):
                if self.cache:
                    if self.fs["client"].exists(index_path_cache):
                        log.debug("Loading index from cache")
                        loadfromcache(index_path_cache)
                    else:
                        log.debug("Loading index from scratch and saving to cache")
                        download(nrows=nrows)
                        save2cache(index_path_cache)
                else:
                    log.debug("Loading index from scratch")
                    download(nrows=nrows)

        if self.N_RECORDS == 0:
            raise DataNotFound("No data found in the index")
        elif nrows is not None and self.N_RECORDS != nrows:
            self.index = self.index[0 : nrows - 1]

        return self

    def run(self, nrows=None):
        """Filter index with search criteria"""

        def search2cache_path(path, nrows=None):
            if nrows is not None:
                cache_path = path + "/local" + "#%i.%s" % (nrows, self.ext)
            else:
                cache_path = path + "/local.%s" % self.ext
            return cache_path

        search_path_cache = search2cache_path(self.search_path, nrows=nrows)

        if self.cache and self.fs["client"].exists(
            search_path_cache
        ):  # and self._same_origin(search_path_cache):
            log.debug(
                "Search results already in memory as a Pyarrow table, loading from '%s'"
                % search_path_cache
            )
            self.search = self._read(
                self.fs["client"].fs, search_path_cache, fmt=self.ext
            )
            self.search_path_cache.commit(search_path_cache)
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

            log.debug(
                "Found %i/%i matches" % (self.search.shape[0], self.index.shape[0])
            )
            if self.cache and self.search.shape[0] > 0:
                self._write(
                    self.fs["client"], search_path_cache, self.search, fmt=self.ext
                )
                self.search = self._read(self.fs["client"].fs, search_path_cache)
                self.search_path_cache.commit(search_path_cache)
                log.debug(
                    "Search results saved in cache as a Pyarrow table at '%s'"
                    % search_path_cache
                )
        return self

    def _to_dataframe(self, nrows=None, index=False):  # noqa: C901
        """Return search results as dataframe

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

        df.drop("longitude_360", inplace=True, axis="columns")
        return df, src

    def _reduce_a_filter_list(self, filters, op="or"):
        if version.parse(pa.__version__) < version.parse("7.0"):
            filters = [i.to_pylist() for i in filters]
        if op == "or":
            return np.logical_or.reduce(filters)
        elif op == "and":
            return np.logical_and.reduce(filters)

    @property
    def search_path(self):
        """Path to search result uri"""
        # return self.host + "/" + self.index_file + "." + self.sha_pq
        return self.fs["client"].fs.sep.join(
            [self.host, "%s.%s" % (self.index_file, self.sha_pq)]
        )

    @property
    def uri_full_index(self) -> List[str]:
        """File paths listed in the index"""
        sep = self.fs["src"].fs.sep
        return [
            sep.join(
                [self.host.replace("/idx", ""), "dac", f.as_py().replace("/", sep)]
            )
            for f in self.index["file"]
        ]

    @property
    def uri(self) -> List[str]:
        """File paths listed in search results"""
        sep = self.fs["src"].fs.sep
        return [
            sep.join(
                [self.host.replace("/idx", ""), "dac", f.as_py().replace("/", sep)]
            )
            for f in self.search["file"]
        ]

    def read_wmo(self, index=False):
        """Return list of unique WMOs from the index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        list(int)
        """
        if hasattr(self, "search") and not index:
            results = pa.compute.split_pattern(self.search["file"], pattern="/")
        else:
            if not hasattr(self, "index"):
                self.load(nrows=self._nrows_index)
            results = pa.compute.split_pattern(self.index["file"], pattern="/")
        df = results.to_pandas()

        def fct(row):
            return row[1]

        wmo = df.map(fct)
        wmo = wmo.unique()
        wmo = [int(w) for w in wmo]
        return wmo

    def read_dac_wmo(self, index=False):
        """Return a tuple of unique [DAC, WMO] pairs from the index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        tuple
        """
        if hasattr(self, "search") and not index:
            results = pa.compute.split_pattern(
                self.search["file"], pattern="/", max_splits=2
            )
        else:
            if not hasattr(self, "index"):
                self.load(nrows=self._nrows_index)
            results = pa.compute.split_pattern(
                self.index["file"], pattern="/", max_splits=2
            )

        results = pa.compute.split_pattern(
            pa.compute.binary_join(
                pa.compute.list_slice(results, start=0, stop=2), "/"
            ).unique(),
            pattern="/",
            max_splits=2,
        ).to_pylist()

        results = tuple(results)
        for ifloat, (dac, wmo) in enumerate(results):
            results[ifloat][1] = int(wmo)

        return results

    def read_params(self, index=False):
        if "parameters" not in self.convention_columns:
            raise InvalidDatasetStructure("Cannot list parameters in this index")
        if hasattr(self, "search") and not index:
            if self.N_MATCH == 0:
                raise DataNotFound(
                    "No data found in the index corresponding to your search criteria."
                    " Search definition: %s" % self.cname
                )
            df = pa.compute.split_pattern(
                self.search["parameters"], pattern=" "
            ).to_pandas()
        else:
            if not hasattr(self, "index"):
                self.load(nrows=self._nrows_index)
            df = pa.compute.split_pattern(
                self.index["parameters"], pattern=" "
            ).to_pandas()
        plist = set(df[0])
        fct = lambda row: len([plist.add(v) for v in row])  # noqa: E731
        df.map(fct)
        return sorted(list(plist))

    def read_domain(self, index=False):
        if "longitude" not in self.convention_columns:
            raise InvalidDatasetStructure("Cannot search for coordinates in this index")
        max = lambda x: pa.compute.max(x).as_py()  # noqa: E731
        min = lambda x: pa.compute.min(x).as_py()  # noqa: E731
        tmin = lambda x: pd.to_datetime(min(x)).to_numpy()  # noqa: E731
        tmax = lambda x: pd.to_datetime(max(x)).to_numpy()  # noqa: E731

        def xmin(xtble):
            if OPTIONS["longitude_convention"] == "360":
                xcol = "longitude_360"
            else:  # OPTIONS['longitude_convention'] == '180':
                xcol = "longitude"
            return min(xtble[xcol])

        def xmax(xtble):
            if OPTIONS["longitude_convention"] == "360":
                xcol = "longitude_360"
            else:  # OPTIONS['longitude_convention'] == '180':
                xcol = "longitude"
            return max(xtble[xcol])

        if hasattr(self, "search") and not index:
            return [
                xmin(self.search),
                xmax(self.search),
                min(self.search["latitude"]),
                max(self.search["latitude"]),
                tmin(self.search["date"]),
                tmax(self.search["date"]),
            ]
        else:
            if not hasattr(self, "index"):
                self.load()
            return [
                xmin(self.index),
                xmax(self.index),
                min(self.index["latitude"]),
                max(self.index["latitude"]),
                tmin(self.index["date"]),
                tmax(self.index["date"]),
            ]

    def read_files(self, index=False) -> List[str]:
        sep = self.fs["src"].fs.sep
        if hasattr(self, "search") and not index:
            return [
                sep.join(["dac", f.as_py().replace("/", sep)])
                for f in self.search["file"]
            ]
        else:
            return [
                sep.join(["dac", f.as_py().replace("/", sep)])
                for f in self.index["file"]
            ]

    def records_per_wmo(self, index=False):
        """Return the number of records per unique WMOs in search results

        Fall back on full index if search not triggered
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
                if not hasattr(self, "index"):
                    self.load(nrows=self._nrows_index)
                search_filter = pa.compute.match_substring_regex(
                    self.index["file"], pattern="/%i/" % wmo
                )
                count[wmo] = self.index.filter(search_filter).shape[0]
        return count

    def to_indexfile(self, file):
        """Save search results on file, following the Argo standard index formats

        Parameters
        ----------
        file: str
            File path to write search results to

        Returns
        -------
        str
        """

        def convert_a_date(row):
            try:
                return row.strftime("%Y%m%d%H%M%S")
            except Exception:
                return ""

        s = self.search

        # Drop internal variable 'longitude_360':
        if "longitude_360" in s.column_names:
            s = s.drop_columns("longitude_360")

        if self.convention not in [
            "ar_index_global_meta",
        ]:
            new_date = pa.array(self.search["date"].to_pandas().apply(convert_a_date))

        new_date_update = pa.array(
            self.search["date_update"].to_pandas().apply(convert_a_date)
        )

        if self.convention not in [
            "ar_index_global_meta",
        ]:
            s = s.set_column(1, "date", new_date)

        if self.convention == "ar_index_global_prof":
            s = s.set_column(7, "date_update", new_date_update)
        elif self.convention in [
            "argo_bio-profile_index",
            "argo_synthetic-profile_index",
        ]:
            s = s.set_column(9, "date_update", new_date_update)
        elif self.convention in ["argo_aux-profile_index"]:
            s = s.set_column(8, "date_update", new_date_update)
        elif self.convention in ["ar_index_global_meta"]:
            s = s.set_column(3, "date_update", new_date_update)

        write_options = csv.WriteOptions(
            delimiter=",", include_header=False, quoting_style="none"
        )
        csv.write_csv(s, file, write_options=write_options)
        file = self._insert_header(file)

        return file

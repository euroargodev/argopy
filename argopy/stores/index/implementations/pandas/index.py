import numpy as np
import pandas as pd
import logging
import gzip
from pathlib import Path
from typing import List

from .....options import OPTIONS
from .....errors import DataNotFound, InvalidDatasetStructure
from .....utils import check_index_cols, conv_lon
from ...spec import ArgoIndexStoreProto


log = logging.getLogger("argopy.stores.index.pd")


class indexstore(ArgoIndexStoreProto):
    """Argo GDAC index store using :class:`pandas.DataFrame` as internal storage format.

    With this store, index and search results are saved as pickle files in cache

    """

    # __doc__ += ArgoIndexStoreProto.__doc__

    backend = "pandas"

    ext = "pd"
    """Storage file extension"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, nrows=None, force=False):  # noqa: C901
        """Load an Argo-index file content

        Returns
        -------
        :class:`pandas.DataFrame`
        """

        def read_csv(input_file, nrows=None):
            this_table = pd.read_csv(
                input_file,
                sep=",",
                index_col=None,
                header=0,
                skiprows=self.skip_rows,
                nrows=nrows,
            )
            return this_table

        def csv2index(obj):
            index = read_csv(obj, nrows=nrows)
            check_index_cols(
                index.columns.to_list(),
                convention=self.convention,
            )
            if "longitude" in self.convention_columns:
                index["longitude_360"] = conv_lon(index["longitude"], "360")
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
                "Argo index file loaded with Pandas read_csv from '%s'"
                % self.index_path
            )
            if self.cache:
                self.fs["src"].fs.save_cache()
            self._nrows_index = nrows

        def save2cache(path_in_cache):
            self._write(self.fs["client"], path_in_cache, self.index, fmt=self.ext)
            self.index = self._read(self.fs["client"], path_in_cache, fmt=self.ext)
            self.index_path_cache = path_in_cache
            log.debug(
                "Argo index saved in cache as a Pandas dataframe at '%s'"
                % path_in_cache
            )

        def loadfromcache(path_in_cache):
            log.debug(
                "Argo index already in cache as a Pandas dataframe, loading from '%s'"
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
            if not hasattr(self, "index"):
                if self.cache:
                    if self.fs["client"].exists(index_path_cache):
                        loadfromcache(index_path_cache)
                    else:
                        download(nrows=nrows)
                        save2cache(index_path_cache)
                else:
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

        if self.cache and self.fs["client"].exists(search_path_cache):
            log.debug(
                "Search results already in cache as a Pandas dataframe, loading from '%s'"
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
                self.search = self.index.head(np.min([nrows, n_match])).reset_index(
                    drop=True
                )
            else:
                self.search = self.index[self.search_filter].reset_index(drop=True)

            log.debug(
                "Found %i/%i matches" % (self.search.shape[0], self.index.shape[0])
            )
            if self.cache and self.search.shape[0] > 0:
                self._write(
                    self.fs["client"], search_path_cache, self.search, fmt=self.ext
                )
                self.search = self._read(
                    self.fs["client"].fs, search_path_cache, fmt=self.ext
                )
                self.search_path_cache.commit(search_path_cache)
                log.debug(
                    "Search results saved in cache as a Pandas dataframe at '%s'"
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
                df = self.search.copy()
        else:
            src = "full index"
            if not hasattr(self, "index"):
                self.load(nrows=nrows)
            df = self.index.copy()

        df.drop("longitude_360", inplace=True, axis="columns")
        return df, src

    def _reduce_a_filter_list(self, filters, op="or"):
        if op == "or":
            return np.logical_or.reduce(filters)
        elif op == "and":
            return np.logical_and.reduce(filters)

    @property
    def search_path(self):
        """Path to search result uri"""
        # return self.host + "/" + self.index_file + "." + self.sha_df
        return self.fs["client"].fs.sep.join(
            [self.host, "%s.%s" % (self.index_file, self.sha_df)]
        )

    @property
    def uri_full_index(self) -> List[str]:
        """File paths listed in the index"""
        sep = self.fs["src"].fs.sep
        return [
            sep.join([self.host.replace("/idx", ""), "dac", f.replace("/", sep)])
            for f in self.index["file"]
        ]

    @property
    def uri(self) -> List[str]:
        """File paths listed in search results"""
        sep = self.fs["src"].fs.sep
        return [
            sep.join([self.host.replace("/idx", ""), "dac", f.replace("/", sep)])
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
            results = self.search["file"].apply(lambda x: int(x.split("/")[1]))
        else:
            results = self.index["file"].apply(lambda x: int(x.split("/")[1]))
        wmo = np.unique(results)
        return wmo

    def read_dac_wmo(self, index=False):
        """Return a tuple of unique [DAC, WMO] pairs from the index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        tuple
        """
        if hasattr(self, "search") and not index:
            results = self.search["file"].apply(lambda x: (x.split("/")[0:2]))
        else:
            results = self.index["file"].apply(lambda x: (x.split("/")[0:2]))
        results = tuple(results.drop_duplicates())
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
            df = self.search["parameters"]
        else:
            if not hasattr(self, "index"):
                self.load()
            df = self.index["parameters"]
        if df.shape[0] > 0:
            plist = set(df[0].split(" "))
            fct = lambda row: len([plist.add(v) for v in row.split(" ")])  # noqa: E731
            df.map(fct)
            return sorted(list(plist))
        else:
            raise DataNotFound("This index is empty")

    def read_domain(self, index=False):
        if "longitude" not in self.convention_columns:
            raise InvalidDatasetStructure("Cannot list parameters in this index")
        tmin = lambda x: pd.to_datetime(str(int(x.min()))).to_numpy()  # noqa: E731
        tmax = lambda x: pd.to_datetime(str(int(x.max()))).to_numpy()  # noqa: E731

        def xmin(xtble):
            if OPTIONS["longitude_convention"] == "360":
                xcol = "longitude_360"
            else:  # OPTIONS['longitude_convention'] == '180':
                xcol = "longitude"
            return xtble[xcol].min()

        def xmax(xtble):
            if OPTIONS["longitude_convention"] == "360":
                xcol = "longitude_360"
            else:  # OPTIONS['longitude_convention'] == '180':
                xcol = "longitude"
            return xtble[xcol].max()

        if hasattr(self, "search") and not index:
            return [
                xmin(self.search),
                xmax(self.search),
                self.search["latitude"].min(),
                self.search["latitude"].max(),
                tmin(self.search["date"]),
                tmax(self.search["date"]),
            ]
        else:
            if not hasattr(self, "index"):
                self.load()
            return [
                xmin(self.index),
                xmax(self.index),
                self.index["latitude"].min(),
                self.index["latitude"].max(),
                tmin(self.index["date"]),
                tmax(self.index["date"]),
            ]

    def read_files(self, index=False) -> List[str]:
        """File paths listed in index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        list(str)
        """
        sep = self.fs["src"].fs.sep
        if hasattr(self, "search") and not index:
            return [sep.join(["dac", f.replace("/", sep)]) for f in self.search["file"]]
        else:
            return [sep.join(["dac", f.replace("/", sep)]) for f in self.index["file"]]

    def records_per_wmo(self, index=False):
        """Return the number of records per unique WMOs in search results

            Fall back on full index if search not triggered

        Returns
        -------
        dict
        """
        ulist = self.read_wmo()
        count = {}
        for wmo in ulist:
            if hasattr(self, "search") and not index:
                search_filter = self.search["file"].str.contains(
                    "/%i/" % wmo, regex=True, case=False
                )
                count[wmo] = self.search[search_filter].shape[0]
            else:
                search_filter = self.index["file"].str.contains(
                    "/%i/" % wmo, regex=True, case=False
                )
                count[wmo] = self.index[search_filter].shape[0]
        return count

    def to_indexfile(self, outputfile):
        """Save search results on file, following the Argo standard index formats

        Parameters
        ----------
        file: str
            File path to write search results to

        Returns
        -------
        str
        """
        df = self.search.copy()

        # Drop internal variable 'longitude_360':
        if "longitude_360" in df.columns:
            df = df.drop("longitude_360", axis=1)

        columns = self.convention_columns
        df.to_csv(
            outputfile,
            sep=",",
            index=False,
            index_label=False,
            header=False,
            columns=columns,
        )
        outputfile = self._insert_header(outputfile)

        return outputfile

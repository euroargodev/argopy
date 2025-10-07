import copy
import numpy as np
import pandas as pd
import xarray as xr
import logging
import time
from abc import ABC, abstractmethod
from fsspec.core import split_protocol
from urllib.parse import urlparse
from typing import Union, List  # noqa: F401
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ...options import OPTIONS
from ...errors import GdacPathError, S3PathError, InvalidDataset, OptionValueError
from ...utils import isconnected, has_aws_credentials
from ...utils import Registry
from ...utils import Chunker
from ...utils import shortcut2gdac

from .. import httpstore, memorystore, filestore, ftpstore, s3store
from .implementations.index_s3 import get_a_s3index
from .implementations.plot import ArgoIndexPlot

try:
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401
except ModuleNotFoundError:
    pass


log = logging.getLogger("argopy.stores.index")


class ArgoIndexStoreProto(ABC):
    backend = "?"
    """Name of store backend (pandas or pyarrow)"""  # Pandas or Pyarrow

    search_type = {}
    """Dictionary with search meta-data"""

    ext = None
    """Storage file extension"""

    convention_supported = [
        "ar_index_global_prof",
        "core",
        "argo_bio-profile_index",
        "bgc-b",
        "bio",
        "argo_synthetic-profile_index",
        "bgc-s",
        "synth",
        "argo_aux-profile_index",
        "aux",
        "ar_index_global_meta",
        "meta",
    ]
    """List of supported conventions"""

    _load_dict = None
    """Place holder for load_dict method"""

    def __init__(
        self,
        host: str = None,
        index_file: str = "ar_index_global_prof.txt",
        convention: str = None,
        cache: bool = False,
        cachedir: str = "",
        timeout: int = 0,
        **kwargs,
    ):
        """Create an Argo index store

        Parameters
        ----------
        host: str, optional, default=OPTIONS["gdac"]
            Local or remote (http, ftp or s3) path to a `dac` folder (compliant with GDAC structure).

            This parameter takes values like:

            - ``http`` or ``https`` for ``https://data-argo.ifremer.fr``
            - ``us-http`` or ``us-https`` for ``https://usgodae.org/pub/outgoing/argo``
            - ``ftp`` for ``ftp://ftp.ifremer.fr/ifremer/argo``
            - ``s3`` or ``aws`` for ``s3://argo-gdac-sandbox/pub/idx``
            - a local absolute path

        index_file: str, default: ``ar_index_global_prof.txt``
            Name of the csv-like text file with the index.

            This parameter takes values like:

            - ``core``  or ``ar_index_global_prof.txt``
            - ``bgc-b`` or ``argo_bio-profile_index.txt``
            - ``bgc-s`` or ``argo_synthetic-profile_index.txt``
            - ``aux``   or ``etc/argo-index/argo_aux-profile_index.txt``
            - ``meta``  or ``ar_index_global_meta.txt``
            - a local absolute path toward a file following an Argo index convention. When using a local file, you need to set the ``convention`` followed by the file.

        convention: str, default: None
            Set the expected format convention of the index file.

            This is useful when trying to load an index file with a custom name.
            If set to ``None``, we'll try to infer the convention from the ``index_file`` value.

            This parameter takes values like:

            - ``core``  or ``ar_index_global_prof``
            - ``bgc-b`` or ``argo_bio-profile_index``
            - ``bgc-s`` or ``argo_synthetic-profile_index``
            - ``aux``   or ``argo_aux-profile_index``
            - ``meta``  or ``ar_index_global_meta``

        cache : bool, default: False
            Use cache or not.
        cachedir: str, default: OPTIONS['cachedir']
            Folder where to store cached files.
        timeout: int,  default: OPTIONS['api_timeout']
            Time out in seconds to connect to a remote host (ftp or http).
        """
        host = OPTIONS["gdac"] if host is None else shortcut2gdac(host)

        self.host = host
        self.root = host  # Will be used by the .uri properties, this is introduced to deal with index files not on the same root as DAC folders

        # Catchup keyword for the main profile index files:
        if index_file in ["core"]:
            index_file = "ar_index_global_prof.txt"
        elif index_file in ["bgc-s", "synth"]:
            index_file = "argo_synthetic-profile_index.txt"
        elif index_file in ["bgc-b", "bio"]:
            index_file = "argo_bio-profile_index.txt"
        elif index_file in ["aux"]:
            index_file = "etc/argo-index/argo_aux-profile_index.txt"
        elif index_file in ["meta"]:
            index_file = "ar_index_global_meta.txt"
        self.index_file = index_file

        # Default number of commented lines to skip at the beginning of csv index files
        # (this is different for s3 than for ftp/http)
        self.skip_rows = 8

        # Create a File Store to access index file:
        self.cache = cache
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self.fs = {}
        if split_protocol(host)[0] is None:
            self.fs["src"] = filestore(cache=cache, cachedir=cachedir)

        elif split_protocol(self.host)[0] in ["https", "http"]:
            # Only for https://data-argo.ifremer.fr (much faster than the ftp servers)
            self.fs["src"] = httpstore(
                cache=cache, cachedir=cachedir, timeout=timeout, size_policy="head"
            )

        elif "ftp" in split_protocol(self.host)[0]:
            if "ifremer" not in host:
                log.info(
                    """Working with a non-official Argo ftp server: %s. Raise on issue if you wish to add your own to the valid list of FTP servers: https://github.com/euroargodev/argopy/issues/new?title=New%%20FTP%%20server"""
                    % host
                )
            if not isconnected(host):
                raise GdacPathError("This host (%s) is not alive !" % host)

            self.fs["src"] = ftpstore(
                host=urlparse(self.host).hostname,  # host eg: ftp.ifremer.fr
                port=0 if urlparse(self.host).port is None else urlparse(self.host).port,
                cache=cache,
                cachedir=cachedir,
                timeout=self.timeout,
                block_size=1000 * (2**20),
            )

        elif "s3" in split_protocol(self.host)[0]:
            # On AWS S3, index files are not under DAC root:
            if self.host == 's3://argo-gdac-sandbox/pub/idx':
                self.root = 's3://argo-gdac-sandbox/pub'
            if self.host == 's3://argo-gdac-sandbox/pub':
                self.host = 's3://argo-gdac-sandbox/pub/idx'
                self.root = 's3://argo-gdac-sandbox/pub'

            if "argo-gdac-sandbox/pub/idx" not in self.host:
                log.info(
                    """Working with a non-official Argo s3 server: %s. Raise on issue if you wish to add your own to the valid list of S3 servers: https://github.com/euroargodev/argopy/issues/new?title=New%%20S3%%20server"""
                    % self.host
                )
            if not isconnected(self.host):
                raise S3PathError("This host (%s) is not alive !" % self.host)

            self.fs["src"] = s3store(
                cache=cache,
                cachedir=cachedir,
                anon=not has_aws_credentials(),
            )
            self.skip_rows = 10

        else:
            raise GdacPathError(
                "Unknown protocol for an Argo index store: %s" % split_protocol(host)[0]
            )

        # Create a File Store to manage search results:
        self.fs["client"] = memorystore(cache, cachedir, skip_instance_cache=True)

        # Registry to Track files opened with the memory store
        # (since it's a global store, other instances will access the same fs, we need our registry here)
        self._memory_store_content = Registry(name="memory store")

        # Registry to Track cached files related to search:
        self.search_path_cache = Registry(name="cached search")

        # Try to infer index convention from the file name:
        if convention is None:
            convention = index_file.split(self.fs["src"].fs.sep)[-1].split(".")[0]
        if convention not in self.convention_supported:
            raise OptionValueError(
                "Convention '%s' is not supported, it must be one in: %s"
                % (convention, self.convention_supported)
            )
        else:
            # Catch shortcuts for convention:
            if convention in ["core"]:
                convention = "ar_index_global_prof"
            elif convention in ["bgc-s", "synth"]:
                convention = "argo_synthetic-profile_index"
            elif convention in ["bgc-b", "bio"]:
                convention = "argo_bio-profile_index"
            elif convention in ["aux"]:
                convention = "argo_aux-profile_index"
            elif convention in ["meta"]:
                convention = "ar_index_global_meta"
        self._convention = convention

        # Check if the index file exists
        # Allow for up to 10 try to account for some slow servers
        i_try, max_try, index_found = 0, 1 if "invalid" in self.host else 10, False
        while i_try < max_try:
            if not self.fs["src"].exists(self.index_path) and not self.fs["src"].exists(
                self.index_path + ".gz"
            ):
                time.sleep(1)
                i_try += 1
            else:
                index_found = True
                break
        if not index_found:
            raise GdacPathError("Index file does not exist: %s" % self.index_path)
        else:
            # Will init search with full index by default:
            self._nrows_index = None
            # Work with the compressed index if available:
            if self.fs["src"].exists(self.index_path + ".gz"):
                self.index_file += ".gz"

        if isinstance(self.fs["src"], s3store):
            # If the index host is on a S3 store, we add another file system that will be called to
            # bypass some search methods to improve performances.
            self.fs["s3"] = get_a_s3index(self.convention)
            # Adjust S3 bucket name and key with host and index file names:
            self.fs["s3"].bucket_name = Path(split_protocol(self.host)[1]).parts[0]
            self.fs["s3"].key = str(
                Path(*Path(split_protocol(self.host)[1]).parts[1:]) / self.index_file
            )

        # # CNAME internal manager to be able to chain search methods:
        # self._cname = None

    def __repr__(self):
        summary = ["<argoindex.%s>" % self.backend]
        summary.append("Host: %s" % self.host)
        summary.append("Index: %s" % self.index_file)
        summary.append("Convention: %s (%s)" % (self.convention, self.convention_title))
        if hasattr(self, "index"):
            summary.append("In memory: True (%i records)" % self.N_RECORDS)
        elif "s3" in self.host:
            summary.append(
                "In memory: False [But there's no need to load the full index with a S3 host to make wmo/cycles searches]"
            )
        else:
            summary.append("In memory: False")

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
        """string formatting helper"""
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
    def index_path(self):
        """Absolute path to the index file"""
        return self.fs["src"].fs.sep.join([self.host, self.index_file])

    @property
    def cname(self) -> str:
        """Search query as a pretty formatted string

        Return 'full' if a search was not yet performed on the :class:`ArgoIndex` instance

        This method uses the BOX, WMO, CYC keys of the index instance ``search_type`` property
        """
        cname = "full"
        C = []

        for key in self.search_type.keys():

            if key == "LAT":
                LAT = self.search_type["LAT"]
                cname = ("y=%0.2f/%0.2f") % (
                    LAT[0],
                    LAT[1],
                )

            elif key == "LON":
                LON = self.search_type["LON"]
                cname = ("x=%0.2f/%0.2f") % (
                    LON[0],
                    LON[1],
                )

            elif key == "DATE":
                DATE = self.search_type["DATE"]
                cname = ("t=%s/%s") % (
                    self._format(DATE[0], "tim"),
                    self._format(DATE[1], "tim"),
                )

            elif key == "BOX":
                BOX = self.search_type["BOX"]
                cname = ("x=%0.2f/%0.2f;y=%0.2f/%0.2f;t=%s/%s") % (
                    BOX[0],
                    BOX[1],
                    BOX[2],
                    BOX[3],
                    self._format(BOX[4], "tim"),
                    self._format(BOX[5], "tim"),
                )

            elif "WMO" == key:
                WMO = self.search_type["WMO"]
                if len(WMO) == 1:
                    cname = "WMO%i" % (WMO[0])
                else:
                    cname = ";".join(["WMO%i" % wmo for wmo in sorted(WMO)])

            elif "CYC" == key:
                CYC = self.search_type["CYC"]
                if len(CYC) == 1:
                    cname = "CYC%i" % (CYC[0])
                else:
                    cname = ";".join(["CYC%i" % cyc for cyc in sorted(CYC)])
                cname = "%s" % cname

            elif "PARAMS" == key:
                PARAM, LOG = self.search_type["PARAMS"]
                cname = ("_%s_" % LOG).join(PARAM)

            elif "DMODE" == key:
                DMODE, LOG = self.search_type["DMODE"]
                cname = ("_%s_" % LOG).join(
                    ["%s_%s" % (p, "".join(DMODE[p])) for p in DMODE]
                )

            elif "PTYPE" == key:
                PTYPE = self.search_type["PTYPE"]
                if len(PTYPE) == 1:
                    cname = "PTYPE%i" % (PTYPE[0])
                else:
                    cname = ";".join(["PTYPE%i" % pt for pt in sorted(PTYPE)])
                cname = "%s" % cname

            elif "PLABEL" == key:
                PLABEL = self.search_type["PLABEL"]
                LOG = 'or'
                cname = ("_%s_" % LOG).join(PLABEL)

            C.append(cname)

        return "_and_".join(C)

    def _sha_from(self, path):
        """Internal post-processing for a sha

        Used by: sha_df, sha_pq, sha_h5
        """
        sha = path  # no encoding
        # sha = hashlib.sha256(path.encode()).hexdigest()  # Full encoding
        # log.debug("%s > %s" % (path, sha))
        return sha

    @property
    def sha_df(self) -> str:
        """Returns a unique SHA for a cname/dataframe"""
        cname = "pd-%s" % self.cname
        sha = self._sha_from(cname)
        return sha

    @property
    def sha_pq(self) -> str:
        """Returns a unique SHA for a cname/parquet"""
        cname = "pq-%s" % self.cname
        # if cname == "full":
        #     raise ValueError("Search not initialised")
        # else:
        #     path = cname
        sha = self._sha_from(cname)
        return sha

    @property
    def sha_h5(self) -> str:
        """Returns a unique SHA for a cname/hdf5"""
        cname = "h5-%s" % self.cname
        # if cname == "full":
        #     raise ValueError("Search not initialised")
        # else:
        #     path = cname
        sha = self._sha_from(cname)
        return sha

    @property
    def _r4(self):
        """Reference table 4 "Argo data centres and institutions" as a dictionary"""
        if self._load_dict is None:
            from ...related import load_dict
            self._load_dict = load_dict
        return self._load_dict('institutions')

    @property
    def _r8(self):
        """Reference table 8 "Argo instrument types" as a dictionary"""
        if self._load_dict is None:
            from ...related import load_dict
            self._load_dict = load_dict
        return self._load_dict('profilers')

    @property
    def shape(self):
        """Shape of the index array"""
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        return self.index.shape

    @property
    def N_FILES(self):
        """Number of rows in search result or index if search not triggered"""
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "search"):
            return self.search.shape[0]
        elif hasattr(self, "index"):
            return self.index.shape[0]
        else:
            raise InvalidDataset("You must, at least, load the index first !")

    @property
    def N_RECORDS(self):
        """Number of rows in the full index"""
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "index"):
            return self.index.shape[0]
        elif "s3" in self.host:
            return np.Inf
        else:
            raise InvalidDataset("Load the index first !")

    @property
    def N_MATCH(self):
        """Number of rows in search result"""
        # Must work for all internal storage type (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        if hasattr(self, "search"):
            return self.search.shape[0]
        else:
            raise InvalidDataset("Initialised search first !")

    @property
    def convention(self):
        """Convention of the index (standard csv file name)"""
        return self._convention

    @property
    def convention_title(self):
        """Long name for the index convention"""
        if self.convention in ["ar_index_global_prof", "core"]:
            title = "Profile directory file of the Argo GDAC"
        elif self.convention in ["argo_bio-profile_index", "bgc-b", "bio"]:
            title = "Bio-Profile directory file of the Argo GDAC"
        elif self.convention in ["argo_synthetic-profile_index", "bgc-s", "synth"]:
            title = "Synthetic-Profile directory file of the Argo GDAC"
        elif self.convention in ["argo_aux-profile_index", "aux"]:
            title = "Aux-Profile directory file of the Argo GDAC"
        elif self.convention in ["ar_index_global_meta", "meta"]:
            title = "Metadata directory file of the Argo GDAC"
        return title

    @property
    def convention_columns(self) -> List[str]:
        """CSV file column names for the index convention"""
        if self.convention == "ar_index_global_prof":
            columns = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
                               'date_update']
        elif self.convention in ["argo_bio-profile_index", "argo_synthetic-profile_index"]:
            columns = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
                               'parameters', 'parameter_data_mode', 'date_update']
        elif self.convention in ["argo_aux-profile_index"]:
            columns = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
                       'parameters', 'date_update']
        elif self.convention in ["ar_index_global_meta"]:
            columns = ['file', 'profiler_type', 'institution', 'date_update']

        return columns

    def _same_origin(self, path):
        """Compare origin of path with current memory fs"""
        return path in self._memory_store_content

    def _commit(self, path):
        self._memory_store_content.commit(path)

    def _write(self, fs, path, obj, fmt="pq"):
        """Write internal array object to file store, possibly cached

        Parameters
        ----------
        fs: Union[filestore, memorystore]
        obj: :class:`pyarrow.Table` or :class:`pandas.DataFrame`
        fmt: str
            File format to use. This is "pq" (default) or "pd"
        """
        this_path = path
        write_this = {
            "pq": lambda o, h: pa.parquet.write_table(o, h),
            "pd": lambda o, h: o.to_pickle(h),  # obj is a pandas dataframe
        }
        if fmt == "parquet":
            fmt = "pq"
        if isinstance(fs, memorystore):
            fs.fs.touch(
                this_path
            )  # Fix for https://github.com/euroargodev/argopy/issues/345
            # fs.fs.touch(this_path)  # Fix for https://github.com/euroargodev/argopy/issues/345
            # This is an f* mystery to me, why do we need 2 calls to trigger file creation FOR REAL ????
            # log.debug("memorystore touched this path before open context: '%s'" % this_path)
        with fs.open(this_path, "wb") as handle:
            write_this[fmt](obj, handle)
            if fs.protocol == "memory":
                self._commit(this_path)
            # log.debug("_write this path: '%s'" % this_path)

        if self.cache:
            fs.fs.save_cache()

        return self

    def _read(self, fs, path, fmt="pq"):
        """Read internal array object from file store

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
            "pq": lambda h: pa.parquet.read_table(h),
            "pd": lambda h: pd.read_pickle(h),
        }
        if fmt == "parquet":
            fmt = "pq"
        with fs.open(this_path, "rb") as handle:
            obj = read_this[fmt](handle)
            # log.debug("_read this path: '%s'" % this_path)
        return obj

    def clear_cache(self) -> Self:
        """Clear cache registry and files associated with this store instance."""
        self.fs["src"].clear_cache()
        self.fs["client"].clear_cache()
        self._memory_store_content.clear()
        self.search_path_cache.clear()
        return self

    def cachepath(self, path):
        """Return path to a cached file

        Parameters
        ----------
        path: str
            Path for which to return the cached file path for. You can use `index` or `search` as shortcuts
            to access path to the internal index or search tables.

        Returns
        -------
        list(str)
        """
        if path == "index" and hasattr(self, "index_path_cache"):
            path = [self.index_path_cache]
        elif path == "search":
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

    def to_dataframe(self, nrows=None, index=False, completed=True):  # noqa: C901
        """Return index or search results as :class:`pandas.DataFrame`

        If search not triggered, fall back on full index by default. Using index=True force to return the full index.

        Parameters
        ----------
        nrows: {int, None}, default: None
            Will return only the first `nrows` of search results. None returns all.
        index: bool, default: False
            Force to return the index, even if a search was performed with this store instance.
        completed: bool, default: True
            Complete the raw index columns with: Platform Number (WMO), Cycle Number, Institution and Profiler details
            This is adding an extra computation, so if you care about performances, you may set this to False.

        Returns
        -------
        :class:`pandas.DataFrame`
        """

        def get_filename(s, index):
            if hasattr(self, "search") and not index:
                fname = s.search_path
            else:
                fname = s.index_path

            if not completed:
                suff = "_raw"
            else:
                suff = ""

            if nrows is not None:
                fname = fname + "/export" + suff + "#%i.pd" % nrows
            else:
                fname = fname + "/export" + suff + ".pd"

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
            from ...related import mapp_dict

            if nrows is not None:
                df = df.loc[0 : nrows - 1].copy()

            if "index" in df:
                df.drop("index", axis=1, inplace=True)

            df.reset_index(drop=True, inplace=True)
            if "date" in df:
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
            df["date_update"] = pd.to_datetime(df["date_update"], format="%Y%m%d%H%M%S")
            df["wmo"] = df["file"].apply(lambda x: int(x.split("/")[1]))
            if self.convention not in [
                "ar_index_global_meta",
            ]:
                df["cyc"] = df["file"].apply(
                    lambda x: int(x.split("_")[1].split(".nc")[0].replace("D", ""))
                )

            if 'profiler_type' in self.convention_columns:
                df['profiler_type'] = df['profiler_type'].fillna(9999).astype(int)

            if completed:
                # institution & profiler mapping for all users
                # todo: may be we need to separate this for standard and expert users
                institution_dictionary = self._r4
                df["tmp1"] = df["institution"].apply(
                    lambda x: mapp_dict(institution_dictionary, x)
                )
                df = df.rename(
                    columns={"institution": "institution_code", "tmp1": "institution"}
                )
                df["dac"] = df["file"].apply(lambda x: x.split("/")[0])

                profiler_dictionary = self._r8

                def ev(x):
                    try:
                        return int(x)
                    except Exception:
                        return x

                df["profiler"] = df["profiler_type"].apply(
                    lambda x: mapp_dict(profiler_dictionary, ev(x))
                )
                df = df.rename(columns={"profiler_type": "profiler_code"})

            if self.cache:
                self._write(self.fs["client"], fname, df, fmt="pd")
                df = self._read(self.fs["client"].fs, fname, fmt="pd")
                if not index:
                    self.search_path_cache.commit(
                        fname
                    )  # Keep track of files related to search results
                log.debug("This dataframe saved in cache. dest='%s'" % fname)

        return df

    def to_indexfile(self):
        """Save search results on file, following the Argo standard index format"""
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def search_path(self):
        """Path to search result uri

        Returns
        -------
        str
        """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri_full_index(self) -> List[str]:
        """List of URI from index

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def uri(self) -> List[str]:
        """List of URI from search results

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @property
    def files(self) -> List[str]:
        """File paths listed in search results"""
        return self.read_files(index=False)

    @property
    def files_full_index(self) -> List[str]:
        """File paths listed in the index"""
        return self.read_files(index=True)

    @property
    def domain(self):
        """Space/time domain of the index

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        return self.read_domain()

    @abstractmethod
    def load(self, nrows=None, force=False):
        """Load an Argo-index file content in memory

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
        """Execute index search query (internal use)

        This method will populate the self.search internal property

        If store is cached, caching is triggered here
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def _to_dataframe(self) -> pd.DataFrame:
        """Return search results as dataframe

        If store is cached, caching is triggered here
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_wmo(self):
        """Return list of unique WMOs in index or search results

        Fall back on full index if search not found

        Returns
        -------
        list(int)
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_dac_wmo(self, index=False):
        """Return a tuple of unique [DAC, WMO] pairs from the index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        tuple
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_params(self):
        """Return list of unique PARAMETERs in index or search results

        Fall back on full index if search not found

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_files(self, index=False):
        """Return file paths listed in index or search results

        Fall back on full index if search not triggered

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_domain(self):
        """Read the space/time domain of the index

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.

        Fall back on full index if search not found

        Returns
        -------
        list
            [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def records_per_wmo(self):
        """Return the number of records per unique WMOs in index or search results

        Fall back on full index if search not found

        Returns
        -------
        dict
            WMO are in keys, nb of records in values
        """
        raise NotImplementedError("Not implemented")

    def _insert_header(self, originalfile):
        if self.convention == "ar_index_global_prof":
            header = """# Title : Profile directory file of the Argo Global Data Assembly Center
# Description : The directory file describes all individual profile files of the argo GDAC ftp site.
# Project : ARGO
# Format version : 2.0
# Date of update : %s
# FTP root number 1 : ftp://ftp.ifremer.fr/ifremer/argo/dac
# FTP root number 2 : ftp://usgodae.org/pub/outgoing/argo/dac
# GDAC node : CORIOLIS
file,date,latitude,longitude,ocean,profiler_type,institution,date_update
""" % pd.to_datetime(
                "now", utc=True
            ).strftime(
                "%Y%m%d%H%M%S"
            )

        elif self.convention == "argo_bio-profile_index":
            header = """# Title : Bio-Profile directory file of the Argo Global Data Assembly Center
# Description : The directory file describes all individual bio-profile files of the argo GDAC ftp site.
# Project : ARGO
# Format version : 2.2
# Date of update : %s
# FTP root number 1 : ftp://ftp.ifremer.fr/ifremer/argo/dac
# FTP root number 2 : ftp://usgodae.org/pub/outgoing/argo/dac
# GDAC node : CORIOLIS
file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
""" % pd.to_datetime(
                "now", utc=True
            ).strftime(
                "%Y%m%d%H%M%S"
            )

        elif self.convention == "argo_synthetic-profile_index":
            header = """# Title : Synthetic-Profile directory file of the Argo Global Data Assembly Center
# Description : The directory file describes all individual synthetic-profile files of the argo GDAC ftp site.
# Project : ARGO
# Format version : 2.2
# Date of update : %s
# FTP root number 1 : ftp://ftp.ifremer.fr/ifremer/argo/dac
# FTP root number 2 : ftp://usgodae.org/pub/outgoing/argo/dac
# GDAC node : CORIOLIS
file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
""" % pd.to_datetime(
                "now", utc=True
            ).strftime(
                "%Y%m%d%H%M%S"
            )

        elif self.convention == "argo_aux-profile_index":
            header = """# Title : Aux-Profile directory file of the Argo Global Data Assembly Center
# Description : The directory file describes all aux-profile files of the argo GDAC ftp site.
# Project : ARGO
# Format version : 2.2
# Date of update : %s
# FTP root number 1 : ftp://ftp.ifremer.fr/ifremer/argo/dac
# FTP root number 2 : ftp://usgodae.org/pub/outgoing/argo/dac
# GDAC node : CORIOLIS
file,date,latitude,longitude,ocean,profiler_type,institution,parameters,date_update
""" % pd.to_datetime(
                "now", utc=True
            ).strftime(
                "%Y%m%d%H%M%S"
            )

        elif self.convention == "ar_index_global_meta":
            header = """# Title : Metadata directory file of the Argo Global Data Assembly Center
# Description : The directory file describes all metadata files of the argo GDAC ftp site.
# Project : ARGO
# Format version : 2.0
# Date of update : %s
# FTP root number 1 : ftp://ftp.ifremer.fr/ifremer/argo/dac
# FTP root number 2 : ftp://usgodae.org/pub/outgoing/argo/dac
# GDAC node : CORIOLIS
file,profiler_type,institution,date_update
""" % pd.to_datetime(
                "now", utc=True
            ).strftime(
                "%Y%m%d%H%M%S"
            )

        with open(originalfile, "r") as f:
            data = f.read()

        with open(originalfile, "w") as f:
            f.write(header)
            f.write(data)

        return originalfile

    def _copy(
        self,
        deep: bool = True,
    ) -> Self:
        cls = self.__class__

        if deep:
            # Ensure complete independence between the original and the copied index:
            obj = cls.__new__(cls)
            obj.__init__(
                host=copy.deepcopy(self.host),
                index_file=copy.deepcopy(self.index_file),
                timeout=copy.deepcopy(self.timeout),
                cache=copy.deepcopy(self.cache),
                cachedir=copy.deepcopy(self.cachedir),
            )
            if hasattr(self, "index"):
                obj._nrows_index = copy.deepcopy(self._nrows_index)
                obj.index = copy.deepcopy(self.index)
                if self.cache:
                    obj.index_path_cache = copy.deepcopy(self.index_path_cache)

        else:
            obj = cls.__new__(cls)
            obj.__init__(
                host=copy.copy(self.host),
                index_file=copy.copy(self.index_file),
                timeout=copy.copy(self.timeout),
                cache=copy.copy(self.cache),
                cachedir=copy.copy(self.cachedir),
            )
            if hasattr(self, "index"):
                obj._nrows_index = copy.copy(self._nrows_index)
                obj.index = copy.copy(self.index)
                if self.cache:
                    obj.index_path_cache = copy.copy(self.index_path_cache)

            if hasattr(self, "search"):
                obj.search_type = copy.copy(self.search_type)
                obj.search_filter = copy.copy(self.search_filter)
                obj.search = copy.copy(self.search)
                if obj.cache:
                    obj.search_path_cache = copy.copy(self.search_path_cache)

        return obj

    def __copy__(self) -> Self:
        return self._copy(deep=False)

    def __deepcopy__(self) -> Self:
        return self._copy(deep=True)

    def copy(
        self,
        deep: bool = True,
    ) -> Self:
        """Returns a copy of this :class:`ArgoIndex` instance

        A copy is a new instance based on similar parameters (e.g. ``host`` and ``index_file``).

        A deep copy ensure complete independence between the original and the copied index.
        If the index was loaded, a new view is returned with the copied index, but search parameters and results are lost.

        A shallow copy preserves the index array, search parameters and results.

        Parameters
        ----------
        deep: bool, optional, default=True

            Whether the search parameters and results are copied onto the new ArgoIndex instance.

        Returns
        -------
        :class:`ArgoIndex`
        """
        return self._copy(deep=deep)

    def iterfloats(self, index=False, chunksize: int = None):
        """Iterate over unique Argo floats in the full index or search results

        By default, iterate over a single float, otherwise use the `chunksize` argument to iterate over chunk of floats.

        Parameters
        ----------
        index: bool, optional, default=False
            Passed to :class:`ArgoIndex.read_wmo` in order to choose if we iterate over all WMOs of the index or
            only those matching search results.

        chunksize: int, optional
            Maximum chunk size

            Eg: A value of 5 will create chunks with as many as 5 WMOs each.

        Returns
        -------
        Iterator of :class:`ArgoFloat`

        Examples
        --------
        .. code-block:: python
            :caption: Example of iteration

            # Make a search on Argo index of profiles:
            idx = ArgoIndex().search_lat_lon([lon_min, lon_max, lat_min, lat_max])

            # Then iterate over float matching the results:
            for float in idx.iterfloats():
                float # is a ArgoFloat instance

        """
        from .. import ArgoFloat  # Prevent circular import

        wmos = self.read_wmo(index=index)

        if chunksize is not None:
            chk_opts = {}
            chk_opts.update({'chunks': {'wmo': 'auto'}})
            chk_opts.update({'chunksize': {'wmo': chunksize}})
            chunked = Chunker({'wmo': self.read_wmo(index=index)}, **chk_opts).fit_transform()
            for grp in chunked:
                yield [ArgoFloat(wmo, idx=self) for wmo in grp]

        else:
            for wmo in wmos:
                yield ArgoFloat(wmo, idx=self)

    # Possibly register extensions without specific implementations:
    plot = xr.core.utils.UncachedAccessor(ArgoIndexPlot)

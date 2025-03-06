import io
import pandas as pd
import logging
from decorator import decorator
import warnings

from ....utils.checkers import (
    check_index_cols,
    check_wmo,
    check_cyc,
    is_list_of_strings,
    has_aws_credentials,
    HAS_BOTO3,
)
from ....utils import redact
from ....errors import InvalidDatasetStructure
from ... import s3store

try:
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa

    HAS_PYARROW = True
except ModuleNotFoundError:
    HAS_PYARROW = False

    class pa:
        @property
        def Table(self):
            pass

    pass

if HAS_BOTO3:
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config


log = logging.getLogger("argopy.stores.index.s3")


@decorator
def requires_pyarrow(func, *args, **kwargs):
    if not HAS_PYARROW:
        warnings.warn(
            "The 'pyarrow' library is not installed. Please install it to ensure full functionality."
        )
    return func(*args, **kwargs)


class s3index:
    """
    A prototype for an Argo index store relying on remote CSV or PQ index data on S3.

    The index is scanned/searched directly on the s3 server using the :class:`boto3.client.select_object_content` boto3 method.

    The key limitation here is that we can only search for WMO and CYC.
    This is due to the fact that G. MAZE could not manage to convert (CAST) latitude, longitude and time to a more
    appropriate data type to execute a search. All variables are considered string by default.

    Examples
    --------
    idx = s3index()

    idx.search_wmo(6903091)
    idx.search_wmo(6903091)
    idx.search_wmo([13857, 6903091])
    idx.search_cyc(1)
    idx.search_cyc(1, nrows=100)
    idx.search_cyc([0, 1])
    idx.search_cyc([0, 1], nrows=100)
    idx.search_wmo_cyc(6903091, 1)
    idx.search_wmo_cyc(6903091, [1, 2])
    idx.search_wmo_cyc([13857, 6903091], 1)
    idx.search_wmo_cyc([13857, 6903091], [1, 2])

    idx.pd # Return search results as a :class:`pd.DataFrame`
    idx.pq # Return search results as a :class:`pa.Table`

    idx.stats # Data processing stats

    """

    bucket_name = "argo-gdac-sandbox"
    """Name of the S3 bucket"""

    sql_formaters = {
        "search_wmo": {
            "split": "SELECT * FROM s3object s WHERE s._1 LIKE '%/{wmo}/profiles/%'".format,
            "single": "s._1 LIKE '%/{wmo}/profiles/%'".format,
        },
        "search_cyc": {
            "split": "SELECT * FROM s3object s WHERE s._1 LIKE '%/%/profiles/%_{cyc:03d}.nc'".format,
            "single": "s._1 LIKE '%/%/profiles/%_{cyc:03d}.nc'".format,
        },
        "search_wmo_cyc": {
            "split": "SELECT * FROM s3object s WHERE s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format,
            "single": "s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format,
        },
    }
    """SQL syntax formatters for all search methods and for each strategy"""

    def __init__(self):
        # Create a boto3 client to interface with S3
        if has_aws_credentials():
            self.fs = boto3.client("s3")
            try:
                access_key = (
                    self.fs._request_signer._credentials.get_frozen_credentials().access_key
                )
                log.debug(
                    "Found AWS Credentials for access_key='%s'" % redact(access_key, 4)
                )
            except:  # noqa: E722
                pass
        else:
            self.fs = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            log.debug(
                "No AWS Credentials found, running UNSIGNED anonymous boto3 requests"
            )
            # search engines won't be available !

        self.stats_last = {}
        self.stats = {}
        self._sql_logic = "single"  # 'single' or 'split'

    def _sio2pd(self, obj_io: io.StringIO) -> pd.DataFrame:
        def _pd(input_io):
            this_table = pd.read_csv(
                input_io,
                sep=",",
                index_col=None,
                header=None,
                skiprows=0,
                nrows=None,
                names=self.colNames,
            )
            return this_table

        index = _pd(obj_io)
        check_index_cols(
            index.columns.to_list(),
            convention=self.convention,
        )
        obj_io.seek(0)  # Rewind
        return index

    @requires_pyarrow
    def _sio2pq(self, obj_io: io.StringIO) -> pa.Table:
        input_bytes = io.BytesIO(obj_io.read().encode("utf8"))
        if input_bytes.getbuffer().nbytes > 0:
            index = csv.read_csv(
                input_bytes,
                read_options=csv.ReadOptions(
                    use_threads=True, skip_rows=0, column_names=self.colNames
                ),
                convert_options=csv.ConvertOptions(
                    column_types={
                        "date": pa.timestamp("s"),  # , tz="utc"
                        "date_update": pa.timestamp("s"),
                    },
                    timestamp_parsers=["%Y%m%d%H%M%S"],
                ),
            )
        else:
            index = self.empty_pq
        check_index_cols(
            index.column_names,
            convention=self.convention,
        )
        obj_io.seek(0)  # Rewind
        return index

    def query(self, sql_expression: str) -> str:
        # Use SelectObjectContent to filter CSV data before downloading it
        try:
            s3_object = self.fs.select_object_content(
                Bucket=self.bucket_name,
                Key=self.key,
                ExpressionType="SQL",
                Expression=sql_expression,
                InputSerialization={
                    "CSV": {
                        "FileHeaderInfo": "IGNORE",
                        "Comments": "#",
                        "QuoteEscapeCharacter": '"',
                        "RecordDelimiter": "\n",
                        "FieldDelimiter": ",",
                        "QuoteCharacter": '"',
                        "AllowQuotedRecordDelimiter": False,
                    },
                    "CompressionType": self.CompressionType,
                },
                OutputSerialization={"CSV": {}},
            )
        except:  # noqa: E722
            # log.debug(boto3.set_stream_logger('botocore', level='DEBUG'))
            raise

        # Iterate over the filtered CSV data
        records = []
        for event in s3_object["Payload"]:
            if "Records" in event:
                records.append(event["Records"]["Payload"].decode("utf-8"))
            elif "Stats" in event:
                stats = event["Stats"]["Details"]

        self.stats_last = stats
        self.stats.update({sql_expression: stats})

        return "".join(r for r in records)

    def run(self):
        if not is_list_of_strings(self.sql_expression):
            self.sql_expression = [self.sql_expression]

        results = ""
        for sql in self.sql_expression:
            results += self.query(sql)
        self.search = io.StringIO(results)
        return self

    def search_wmo(self, WMOs, nrows=None):
        WMOs = check_wmo(WMOs)

        if self._sql_logic == "split":
            sql = []
            for wmo in WMOs:
                sql.append(self.sql_formaters['search_wmo']['split'](wmo=wmo))

        elif self._sql_logic == "single":
            sql = "SELECT * FROM s3object s WHERE "
            if len(WMOs) > 1:
                sql += " OR ".join([self.sql_formaters['search_wmo']['single'](wmo=wmo) for wmo in WMOs])

            else:
                sql += self.sql_formaters['search_wmo']['single'](wmo=WMOs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    def search_cyc(self, CYCs, nrows=None):
        if self.convention in ["ar_index_global_meta"]:
            raise InvalidDatasetStructure(
                "Cannot search for cycle number in this index"
            )
        CYCs = check_cyc(CYCs)

        if self._sql_logic == "split":
            sql = []
            for cyc in CYCs:
                sql.append(self.sql_formaters['search_cyc']['split'](cyc=cyc))

        elif self._sql_logic == "single":
            sql = "SELECT * FROM s3object s WHERE "
            if len(CYCs) > 1:
                sql += " OR ".join(
                    [
                        self.sql_formaters['search_cyc']['single'](cyc=cyc)
                        for cyc in CYCs
                    ]
                )
            else:
                sql += self.sql_formaters['search_cyc']['single'](cyc=CYCs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    def search_wmo_cyc(self, WMOs, CYCs, nrows=None):
        if self.convention in ["ar_index_global_meta"]:
            raise InvalidDatasetStructure(
                "Cannot search for cycle number in this index"
            )
        WMOs = check_wmo(WMOs)
        CYCs = check_cyc(CYCs)

        if self._sql_logic == "split":
            sql = []
            for wmo in WMOs:
                for cyc in CYCs:
                    sql.append(self.sql_formaters['search_wmo_cyc']['split'](wmo=wmo, cyc=cyc))

        elif self._sql_logic == "single":
            sql = "SELECT * FROM s3object s WHERE "
            if len(WMOs) > 1:
                if len(CYCs) > 1:
                    sql += " OR ".join(
                        [
                            self.sql_formaters['search_wmo_cyc']['single'](wmo=wmo, cyc=cyc)
                            for wmo in WMOs
                            for cyc in CYCs
                        ]
                    )
                else:
                    sql += " OR ".join(
                        [
                            self.sql_formaters['search_wmo_cyc']['single'](wmo=wmo, cyc=CYCs[0])
                            for wmo in WMOs
                        ]
                    )
            else:
                if len(CYCs) > 1:
                    sql += " OR ".join(
                        [
                            self.sql_formaters['search_wmo_cyc']['single'](wmo=WMOs[0], cyc=cyc)
                            for cyc in CYCs
                        ]
                    )
                else:
                    sql += self.sql_formaters['search_wmo_cyc']['single'](wmo=WMOs[0], cyc=CYCs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    @property
    def pd(self) -> pd.DataFrame:
        """Return search result as a :class:`pd.DataFrame`"""
        if not hasattr(self, "search"):
            raise Exception("Execute a search first !")
        else:
            return self._sio2pd(self.search)

    @property
    @requires_pyarrow
    def pq(self) -> pa.Table:
        """Return search result as a :class:`pa.Table`"""
        if not hasattr(self, "search"):
            raise Exception("Execute a search first !")
        else:
            return self._sio2pq(self.search)


class s3index_core(s3index):
    # key, CompressionType = "pub/idx/ar_index_global_prof.txt", "NONE"
    key = "pub/idx/ar_index_global_prof.txt.gz"
    """Path to the index source file"""

    CompressionType = "GZIP"
    """Compression used by the index source file"""

    convention = "ar_index_global_prof"
    """Argo convention of the index source file"""

    colNames = [
        "file",
        "date",
        "latitude",
        "longitude",
        "ocean",
        "profiler_type",
        "institution",
        "date_update",
    ]
    """List of the index column names"""

    @property
    @requires_pyarrow
    def empty_pq(self):
        return pa.Table.from_pydict(
            {
                "file": [],
                "date": [],
                "latitude": [],
                "longitude": [],
                "ocean": [],
                "profiler_type": [],
                "institution": [],
                "date_update": [],
            },
            schema=pa.schema(
                [
                    ("file", pa.string()),
                    ("date", pa.timestamp("s")),
                    ("latitude", pa.float64()),
                    ("longitude", pa.float64()),
                    ("ocean", pa.string()),
                    ("profiler_type", pa.int64()),
                    ("institution", pa.string()),
                    ("date_update", pa.timestamp("s")),
                ]
            ),
        )


class s3index_bgc_bio(s3index):
    key = "pub/idx/argo_bio-profile_index.txt.gz"
    """Path to the index source file"""

    CompressionType = "GZIP"
    """Compression used by the index source file"""

    convention = "argo_bio-profile_index"
    """Argo convention of the index source file"""

    colNames = [
        "file",
        "date",
        "latitude",
        "longitude",
        "ocean",
        "profiler_type",
        "institution",
        "parameters",
        "parameter_data_mode",
        "date_update",
    ]
    """List of the index column names"""

    @property
    @requires_pyarrow
    def empty_pq(self):
        return pa.Table.from_pydict(
            {
                "file": [],
                "date": [],
                "latitude": [],
                "longitude": [],
                "ocean": [],
                "profiler_type": [],
                "institution": [],
                "parameters": [],
                "parameter_data_mode": [],
                "date_update": [],
            },
            schema=pa.schema(
                [
                    ("file", pa.string()),
                    ("date", pa.timestamp("s")),
                    ("latitude", pa.float64()),
                    ("longitude", pa.float64()),
                    ("ocean", pa.string()),
                    ("profiler_type", pa.int64()),
                    ("institution", pa.string()),
                    ("parameters", pa.string()),
                    ("parameter_data_mode", pa.string()),
                    ("date_update", pa.timestamp("s")),
                ]
            ),
        )


class s3index_bgc_synthetic(s3index_bgc_bio):
    key = "pub/idx/argo_synthetic-profile_index.txt.gz"
    """Path to the index source file"""

    CompressionType = "GZIP"
    """Compression used by the index source file"""

    convention = "argo_synthetic-profile_index"
    """Argo convention of the index source file"""


class s3index_meta(s3index):
    key = "pub/idx/ar_index_global_meta.txt.gz"
    """Path to the index source file"""

    CompressionType = "GZIP"
    """Compression used by the index source file"""

    convention = "ar_index_global_meta"
    """Argo convention of the index source file"""

    colNames = [
        "file",
        "profiler_type",
        "institution",
        "date_update",
    ]
    """List of the index column names"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sql_formaters.update({
            "search_wmo": {
                "split": "SELECT * FROM s3object s WHERE s._1 LIKE '%/{wmo}/{wmo}_meta.nc'".format,
                "single": "s._1 LIKE '%/{wmo}/{wmo}_meta.nc'".format,
            },
        })

    @property
    @requires_pyarrow
    def empty_pq(self):
        return pa.Table.from_pydict(
            {
                "file": [],
                "profiler_type": [],
                "institution": [],
                "date_update": [],
            },
            schema=pa.schema(
                [
                    ("file", pa.string()),
                    ("profiler_type", pa.int64()),
                    ("institution", pa.string()),
                    ("date_update", pa.timestamp("s")),
                ]
            ),
        )


def get_a_s3index(convention):
    if convention == "ar_index_global_prof":
        return s3index_core()
    elif convention == "argo_bio-profile_index":
        return s3index_bgc_bio()
    elif convention == "argo_synthetic-profile_index":
        return s3index_bgc_synthetic()
    elif convention == "ar_index_global_meta":
        return s3index_meta()


@decorator
def search_s3(func, *args, **kwargs):
    """Decorator for ArgoIndexSearchEngine instance methods patched for S3 store

    This decorator will bypass :class:`argopy.stores.indexstore` search methods with a boto3
    sql request design when using the S3 Argo index store.

    Note that search methods are bypassed only if the index was not loaded before, otherwise we're using the store
    original method working with the internal index structure (pandas dataframe or pyarrow table).
    """
    idx = args[0]._obj

    if (
        func.__name__ == "wmo"
        and not hasattr(idx, "index")
        and isinstance(idx.fs["src"], s3store)
    ):
        WMOs, nrows, composed = args[1], args[2], args[3]
        if not composed:
            WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
            log.debug(
                "Argo index searching for WMOs=[%s] using boto3 SQL request ..."
                % ";".join([str(wmo) for wmo in WMOs])
            )
            idx.fs["s3"].search_wmo(WMOs, nrows=nrows)
            idx.search_type = {"WMO": WMOs}
            idx.search_filter = idx.fs["s3"].sql_expression
            idx.search = getattr(idx.fs["s3"], idx.ext)
            return idx
        else:
            log.debug("Argo index searching using boto3 SQL request not available for composition")

    if (
        func.__name__ == "cyc"
        and not hasattr(idx, "index")
        and isinstance(idx.fs["src"], s3store)
    ):
        CYCs, nrows, composed = args[1], args[2], args[3]
        if not composed:
            CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
            log.debug(
                "Argo index searching for CYCs=[%s] using boto3 SQL request ..."
                % (";".join([str(cyc) for cyc in CYCs]))
            )
            idx.fs["s3"].search_cyc(CYCs, nrows=nrows)
            idx.search_type = {"CYC": CYCs}
            idx.search_filter = idx.fs["s3"].sql_expression
            idx.search = getattr(idx.fs["s3"], idx.ext)
            return idx
        else:
            log.debug("Argo index searching using boto3 SQL request not available for composition")

    if (
        func.__name__ == "wmo_cyc"
        and not hasattr(idx, "index")
        and isinstance(idx.fs["src"], s3store)
    ):
        WMOs, CYCs, nrows, composed = args[1], args[2], args[3], args[4]
        if not composed:
            WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
            CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
            log.debug(
                "Argo index searching for WMOs=[%s] and CYCs=[%s] using boto3 SQL request ..."
                % (
                    ";".join([str(wmo) for wmo in WMOs]),
                    ";".join([str(cyc) for cyc in CYCs]),
                )
            )
            idx.fs["s3"].search_wmo_cyc(WMOs, CYCs, nrows=nrows)
            idx.search_type = {"WMO": WMOs, "CYC": CYCs}
            idx.search_filter = idx.fs["s3"].sql_expression
            idx.search = getattr(idx.fs["s3"], idx.ext)
            return idx
        else:
            log.debug("Argo index searching using boto3 SQL request not available for composition")

    return func(*args, **kwargs)

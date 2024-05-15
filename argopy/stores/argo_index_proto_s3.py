import io
import pandas as pd
from ..utils.checkers import check_index_cols, check_wmo, check_cyc, is_list_of_strings

try:
    import boto3
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa
except ModuleNotFoundError:
    pass

class s3index_for_core:
    """
    A prototype for an Argo index store relying on CSV or PQ index data on S3
    The index is scanned/searched directly on the s3 server using the boto3 client select_object_content method

    The key limitation here is that we can only search for WMO and CYC.
    This is due to the fact that I could not manage to convert (CAST) latitude, longitude and time to a more
    appropriate data type to execute a search. All variables are considered string by default.

    Examples
    --------
    idx = s3index_for_core()

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

    idx.pd # Return search results as :class:`pd.DataFrame`
    idx.pq # Return search results as a :class:`pa.Table`

    idx.stats # Data processing stats

    """

    bucket_name = "argo-gdac-sandbox"
    """Name of the S3 bucket"""

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

    def __init__(self):
        self.fs = boto3.client("s3")  # Create a boto3 client to interface with S3
        self.stats = {}
        self._sql_logic = 'unique'

    def _sio2pd(self, obj_io: io.StringIO) -> pd.DataFrame:
        def _pd(input_io):
            this_table = pd.read_csv(input_io,
                                     sep=",",
                                     index_col=None,
                                     header=None,
                                     skiprows=0,
                                     nrows=None,
                                     names=self.colNames
                                     )
            return this_table

        index = _pd(obj_io)
        check_index_cols(
            index.columns.to_list(),
            convention=self.convention,
        )
        obj_io.seek(0)  # Rewind
        return index

    def _sio2pq(self, obj_io: io.StringIO) -> pa.Table:
        input_bytes = io.BytesIO(obj_io.read().encode('utf8'))
        index = csv.read_csv(
            input_bytes,
            read_options=csv.ReadOptions(use_threads=True,
                                         skip_rows=0,
                                         column_names=self.colNames
                                         ),
            convert_options=csv.ConvertOptions(
                column_types={
                    "date": pa.timestamp("s"),  # , tz="utc"
                    "date_update": pa.timestamp("s"),
                },
                timestamp_parsers=["%Y%m%d%H%M%S"],
            ),
        )
        check_index_cols(
            index.column_names,
            convention=self.convention,
        )
        obj_io.seek(0)  # Rewind
        return index

    def query(self, sql_expression: str) -> str:
        # Use SelectObjectContent to filter the CSV data before downloading it
        s3_object = self.fs.select_object_content(
            Bucket=self.bucket_name,
            Key=self.key,
            ExpressionType="SQL",
            Expression=sql_expression,
            # InputSerialization={"CSV": {"FileHeaderInfo": "USE"}, "CompressionType": "NONE"},
            InputSerialization={"CSV": {"FileHeaderInfo": "IGNORE",
                                        'Comments': '#',
                                        'QuoteEscapeCharacter': '"',
                                        'RecordDelimiter': '\n',
                                        'FieldDelimiter': ',',
                                        'QuoteCharacter': '"',
                                        'AllowQuotedRecordDelimiter': False
                                        },
                                "CompressionType": self.CompressionType},
            OutputSerialization={"CSV": {}},
        )

        # Iterate over the filtered CSV data
        records = []
        for event in s3_object["Payload"]:
            if "Records" in event:
                records.append(event["Records"]["Payload"].decode("utf-8"))
            elif 'Stats' in event:
                stats = event['Stats']['Details']

        self.stats.update({sql_expression: stats})

        return ''.join(r for r in records)

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

        if self._sql_logic == 'split':
            sql = []
            for wmo in WMOs:
                sql.append("SELECT * FROM s3object s WHERE s._1 LIKE '%/{wmo}/profiles/%'".format(wmo=wmo))
        else:
            sql = "SELECT * FROM s3object s WHERE "
            if len(WMOs) > 1:
                sql += " OR ".join(["s._1 LIKE '%/{wmo}/profiles/%'".format(wmo=wmo) for wmo in WMOs])
            else:
                sql += "s._1 LIKE '%/{wmo}/profiles/%'".format(wmo=WMOs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    def search_cyc(self, CYCs, nrows=None):
        CYCs = check_cyc(CYCs)

        if self._sql_logic == 'split':
            sql = []
            for cyc in CYCs:
                sql.append("SELECT * FROM s3object s WHERE s._1 LIKE '%/%/profiles/%_{cyc:03d}.nc'".format(cyc=cyc))
        else:
            sql = "SELECT * FROM s3object s WHERE "
            if len(CYCs) > 1:
                sql += " OR ".join(["s._1 LIKE '%/%/profiles/%_{cyc:03d}.nc'".format(cyc=cyc) for cyc in CYCs])
            else:
                sql += "s._1 LIKE '%/%/profiles/%_{cyc:03d}.nc'".format(cyc=CYCs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    def search_wmo_cyc(self, WMOs, CYCs, nrows=None):
        WMOs = check_wmo(WMOs)
        CYCs = check_cyc(CYCs)

        if self._sql_logic == 'split':
            sql = []
            for wmo in WMOs:
                for cyc in CYCs:
                    sql.append("SELECT * FROM s3object s WHERE s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format(wmo=wmo, cyc=cyc))
        else:
            sql = "SELECT * FROM s3object s WHERE "
            if len(WMOs) > 1:
                if len(CYCs) > 1:
                    sql += " OR ".join(
                        ["s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format(wmo=wmo, cyc=cyc) for wmo in WMOs for cyc in
                         CYCs])
                else:
                    sql += " OR ".join(
                        ["s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format(wmo=wmo, cyc=CYCs[0]) for wmo in WMOs])
            else:
                if len(CYCs) > 1:
                    sql += " OR ".join(
                        ["s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format(wmo=WMOs[0], cyc=cyc) for cyc in CYCs])
                else:
                    sql += "s._1 LIKE '%/{wmo}/profiles/%_{cyc:03d}.nc'".format(wmo=WMOs[0], cyc=CYCs[0])

        if nrows is not None:
            sql += " LIMIT %i" % nrows

        self.sql_expression = sql
        self.run()
        return self

    @property
    def pd(self) -> pd.DataFrame:
        """Return search result as a :class:`pd.DataFrame`"""
        if not hasattr(self, 'search'):
            raise Exception('Execute a search first !')
        else:
            return self._sio2pd(self.search)

    @property
    def pq(self) -> pa.Table:
        """Return search result as a :class:`pa.Table`"""
        if not hasattr(self, 'search'):
            raise Exception('Execute a search first !')
        else:
            return self._sio2pq(self.search)

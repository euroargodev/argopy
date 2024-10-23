"""
An experimental data fetcher for WHOI-produced parquet files

Compatible with local or s3 file systems

"""
import pyarrow.parquet as pq
import dask.dataframe as dd
import xarray as xr
import pandas as pd

from ..stores import s3store, filestore
from ..utils import cast_Argo_variable_type, check_wmo, check_cyc
from .proto import ArgoDataFetcherProto


class toto(ArgoDataFetcherProto):
    def to_xarray(self, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError("Not implemented")

    def transform_data_mode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError("Not implemented")

    def filter_data_mode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError("Not implemented")

    def filter_qc(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError("Not implemented")

    def filter_researchmode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError("Not implemented")



class GDACParquetFetcher:
    dataset_id = "bgc-s"

    def __init__(self, location: str = "local"):

        if location == "local":
            self.parquet_dir = "/Users/gmaze/data/ARGO/parquet/ArgoBGC/"
            self.schema_path = "/Users/gmaze/data/ARGO/parquet/ArgoBGC_schema.metadata"
            self.fs = filestore().fs
        elif location == "s3":
            self.parquet_dir = "s3://argotests/ArgoBGC/"
            self.schema_path = "s3://argotests/ArgoBGC_schema.metadata"
            self.fs = s3store().fs

        self.schema = pq.read_schema(self.schema_path)

        if self.dataset_id == "phy":
            self.definition = "WHOI parquet Argo data fetcher"
        elif self.dataset_id in ["bgc", "bgc-s"]:
            self.definition = "WHOI parquet Argo BGC data fetcher"

        self.filters = None

    def validate_columns(self, columns):
        if columns is None:
            return None
        else:
            return [col for col in columns if col in self.schema.names]

    def to_dataframe(self, columns=None):
        opts = {"engine": "pyarrow", "schema": self.schema}
        # opts['storage_options'] = {"anon": True, "use_ssl": True}
        opts["filesystem"] = self.fs

        if self.filters is not None:
            opts["filters"] = self.filters
        if columns is not None:
            opts["columns"] = self.validate_columns(columns)

        return dd.read_parquet(self.parquet_dir, **opts)

    @property
    def N_PROF(self):
        ddf = self.to_dataframe(columns=["PLATFORM_NUMBER", "CYCLE_NUMBER"])
        return (
            ddf.groupby(["PLATFORM_NUMBER", "CYCLE_NUMBER"]).count().compute().shape[0]
        )

    def to_xarray(self, columns=None, compute_ddf=False):
        this_ddf = self.to_dataframe(columns=columns)
        if compute_ddf:
            this_ddf.compute()

            # Now convert the Dask dataframe into a xarray dataset:
        indexname = "N_POINTS"
        this_ds = xr.Dataset()
        this_ds[indexname] = this_ddf.index

        def to_dask_array(col):
            """Convert one Dask dataframe column (aka Series) into a Dask Array"""
            return (col, this_ddf[col].to_dask_array().compute_chunk_sizes())

        futures = client.map(to_dask_array, this_ddf.columns)
        results = client.gather(futures)

        for x, y in results:
            this_ds[x] = (indexname, y)

        this_ds = cast_Argo_variable_type(this_ds)
        this_ds = this_ds.rename_vars({"JULD": "TIME"})

        return this_ds

    def region(self, box: list):
        self.BOX = box.copy()
        filters = [
            ("LONGITUDE", ">=", self.BOX[0]),
            ("LONGITUDE", "<", self.BOX[1]),
            ("LATITUDE", ">=", self.BOX[2]),
            ("LATITUDE", "<", self.BOX[3]),
            ("PRES", ">=", self.BOX[4]),
            ("PRES", "<", self.BOX[5]),
        ]
        if len(self.BOX) == 8:
            filters.append(("JULD", ">=", pd.to_datetime(self.BOX[6]).to_pydatetime()))
            filters.append(("JULD", "<", pd.to_datetime(self.BOX[7]).to_pydatetime()))

        self.filters = filters
        return self

    def float(self, wmo: int):
        self.WMO = check_wmo(wmo)
        filters = []
        for wmo in self.WMO:
            filters.append([("PLATFORM_NUMBER", "==", wmo)])
        self.filters = filters
        return self

    def profile(self, wmo: int, cyc: int):
        self.WMO = check_wmo(wmo)
        self.CYC = check_cyc(cyc)
        filters = []
        for wmo in self.WMO:
            for cyc in self.CYC:
                filters.append(
                    [("PLATFORM_NUMBER", "==", wmo), ("CYCLE_NUMBER", "==", cyc)]
                )
        self.filters = filters
        return self

from argovis import ArgovisDataFetcher
import pandas as pd
import xarray as xr
adf = ArgovisDataFetcher()
profile = adf.get_profile(3900737, 279)
assert not isinstance(profile, str)
df = adf.to_dataframe([profile])
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape[0] > 0
ds = adf.to_xarray([profile])
assert isinstance(ds, xr.core.dataset.Dataset)

profiles = adf.get_platform_profiles(3900737)
df = adf.to_dataframe(profiles)
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape[0] > 0
_ids = df['_id'].unique()
assert _ids.size > 360
ds = adf.to_xarray(profiles)
assert isinstance(ds, xr.core.dataset.Dataset)

shape = [[[168.6,21.7],[168.6,37.7],[-145.9,37.7],[-145.9,21.7],[168.6,21.7]]]
startDate='2017-9-15'
endDate='2017-9-30'
presRange=[0,50]
selectionProfiles = adf.get_selection_profiles(startDate, endDate, shape, presRange)
df = adf.to_dataframe(selectionProfiles)
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape[0] > 0
_ids = df['_id'].unique()
assert _ids.size >= 200
ds = adf.to_xarray(selectionProfiles)
assert isinstance(ds, xr.core.dataset.Dataset)
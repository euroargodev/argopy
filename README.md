# ![argopy logo](https://avatars1.githubusercontent.com/t/3711886?s=90&v=4) Argo data python library

![build](https://github.com/euroargodev/argopy/workflows/build/badge.svg?branch=feature-fetch-local-ftp)
[![codecov](https://codecov.io/gh/euroargodev/argopy/branch/feature-fetch-local-ftp/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)
[![Requirements Status](https://requires.io/github/euroargodev/argopy/requirements.svg?branch=feature-fetch-local-ftp)](https://requires.io/github/euroargodev/argopy/requirements/?branch=feature-fetch-local-ftp)

``argopy`` is a python library that aims to ease Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators.

Several python packages exist: we are currently in the process of analysing how to merge these libraries toward a single powerfull tool. [List your tool here !](https://github.com/euroargodev/argopy/issues/3)

## Install

Since this is a library in active development, use direct install from this repo to benefit from the last version:

```bash
pip install git+http://github.com/euroargodev/argopy.git@master
```

The ``argopy`` library should work under all OS (Linux, Mac and Windows) and with python versions 3.6, 3.7 and 3.8.

## Usage

Note that the primary data model used to manipulate Argo data is [xarray](https://github.com/pydata/xarray).

### Fetching Argo Data

Init the default data fetcher like:
```python
from argopy import DataFetcher as ArgoDataFetcher
argo_loader = ArgoDataFetcher()
```
and then, request data for a **specific space/time domain**:
```python
ds = argo_loader.region([-85,-45,10.,20.,0,10.]).to_xarray()
ds = argo_loader.region([-85,-45,10.,20.,0,1000.,'2012-01','2014-12']).to_xarray()
```
for **profiles of a given float**: 
```python
ds = argo_loader.profile(6902746, 34).to_xarray()
ds = argo_loader.profile(6902746, np.arange(12,45)).to_xarray()
ds = argo_loader.profile(6902746, [1,12]).to_xarray()
```
or for **one or a collection of floats**:
```python
ds = argo_loader.float(6902746).to_xarray()
ds = argo_loader.float([6902746, 6902747, 6902757, 6902766]).to_xarray()
```

Two Argo data fetchers are available.
1. The Ifremer erddap (recommended, but requires internet connection):
    ```python
    argo_loader = ArgoDataFetcher(backend='erddap')
    ds = argo_loader.profile(6902746, 34).to_xarray()
    ```
1. your own local copy of the GDAC ftp (offline access possible, but more limited than the erddap).
    ```python
    argo_loader = ArgoDataFetcher(backend='localftp', path_ftp='/path/to/your/copy/of/Argo/ftp/dac')
    ds = argo_loader.float(6902746).to_xarray()
    ```

### Data manipulation
Data are returned as a collection of measurements. 
If you want to convert them into a collection of profiles, you can use the xarray accessor named ``argo``:
```python
from argopy import DataFetcher as ArgoDataFetcher
ds = ArgoDataFetcher().float(5903248).to_xarray() # Dimensions: (N_POINTS: 25656)
ds = ds.argo.point2profile() # Dimensions: (N_LEVELS: 71, N_PROF: 368)
```

By default fetched data are returned in memory as [xarray.DataSet](http://xarray.pydata.org/en/stable/data-structures.html#dataset). 
From there, it is easy to convert it to other formats like a [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe):
```python
ds = ArgoDataFetcher().profile(6902746, 34).to_xarray()
df = ds.to_dataframe()
```

or to export it to files:
```python
ds = argo_loader.region([-85,-45,10.,20.,0,1000.]).to_xarray()
ds.to_netcdf('my_selection.nc')
# or by profiles:
ds.argo.point2profile().to_netcdf('my_selection.nc')
```

## Development roadmap

We aim to provide high level helper methods to load Argo data and meta-data from:
- [x] Ifremer erddap
- [x] local copy of the GDAC ftp folder
- [ ] Index files [(ongoing work here)](https://github.com/euroargodev/argopy/pull/7)
- [ ] the argovis dataset [(help wanted here)](https://github.com/euroargodev/argopy/issues/2)
- [ ] any other usefull access point to Argo data ?

We also aim to provide high level helper methods to visualise and plot Argo data and meta-data:
- [ ] Map with trajectories
- [ ] Waterfall plots
- [ ] T/S diagram
- [ ] etc !
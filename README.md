# <img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/> Argo data python library

[![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest/?badge=latest)
![Github Action Status](https://github.com/euroargodev/argopy/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)
[![Requirements Status](https://requires.io/github/euroargodev/argopy/requirements.svg?branch=master)](https://requires.io/github/euroargodev/argopy/requirements/?branch=master)
[![Gitter](https://badges.gitter.im/Argo-floats/argopy.svg)](https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![Profile count](https://img.shields.io/endpoint?style=social&url=https%3A%2F%2Fmap.argo-france.fr%2Fdata%2FARGOFULL.json&logo=data:image/png;base64,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)

``argopy`` is a python library that aims to ease Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators.

Several python packages exist: we are currently in the process of analysing how to merge these libraries toward a single powerfull tool.  
[List your tool here !](https://github.com/euroargodev/argopy/issues/3)

Click here to [![badge](https://img.shields.io/badge/launch-Pangeo%20binder-579ACA.svg?logo=data:image/png;base64,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)](https://binder.pangeo.io/v2/gh/euroargodev/argopy/master?urlpath=lab/tree/docs/tryit.ipynb) and play with ``argopy`` before you even install it (thanks [Pangeo](pangeo.io)).

## Install

Install the last release with pip:
```bash
pip install argopy
```

But since this is a young library in active development, use direct install from this repo to benefit from the lastest version:

```bash
pip install git+http://github.com/euroargodev/argopy.git@master
```

The ``argopy`` library should work under all OS (Linux, Mac and Windows) and with python versions 3.6, 3.7 and 3.8.

## Usage

### Fetching Argo Data

Init the default data fetcher like:
```python
from argopy import DataFetcher as ArgoDataFetcher
argo_loader = ArgoDataFetcher()
```
and then, request data for a **specific space/time domain**:
```python
ds = argo_loader.region([-85,-45,10.,20.,0,10.]).to_xarray()
ds = argo_loader.region([-85,-45,10.,20.,0,1000.,'2012-01','2012-12']).to_xarray()
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
By default fetched data are returned in memory as [xarray.DataSet](http://xarray.pydata.org/en/stable/data-structures.html#dataset). 
From there, it is easy to convert it to other formats like a [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe):
```python
ds = ArgoDataFetcher().profile(6902746, 34).to_xarray()
df = ds.to_dataframe()
```

or to export it to files:
```python
ds = argo_loader.region([-85,-45,10.,20.,0,100.]).to_xarray()
ds.to_netcdf('my_selection.nc')
# or by profiles:
ds.argo.point2profile().to_netcdf('my_selection.nc')
```


### Argo Index Fetcher
Index object is returned as a pandas dataframe.

Init the fetcher:
```python
    from argopy import IndexFetcher as ArgoIndexFetcher

    index_loader = ArgoIndexFetcher()
    index_loader = ArgoIndexFetcher(backend='erddap')    
    #Local ftp backend 
    #index_loader = ArgoIndexFetcher(backend='localftp',path_ftp='/path/to/your/argo/ftp/',index_file='ar_index_global_prof.txt')
```
and then, set the index request index for a domain:
```python
    idx=index_loader.region([-85,-45,10.,20.])
    idx=index_loader.region([-85,-45,10.,20.,'2012-01','2014-12'])
```
or for a collection of floats:
```python
    idx=index_loader.float(6902746)
    idx=index_loader.float([6902746, 6902747, 6902757, 6902766])   
```
then you can see you index as a pandas dataframe or a xarray dataset :
```python
    idx.to_dataframe()
    idx.to_xarray()
```
For plottings methods, you'll need `matplotlib`, `cartopy` and `seaborn` installed (they're not in requirements).  
For plotting the map of your query :
```python    
    idx.plot('trajectory)    
```
![index_traj](https://user-images.githubusercontent.com/17851004/78023937-d0c2d580-7357-11ea-9974-70a2aaf30590.png)

For plotting the distribution of DAC or profiler type of the indexed profiles :
```python    
    idx.plot('dac')    
    idx.plot('profiler')`
```
![dac](https://user-images.githubusercontent.com/17851004/78024137-26977d80-7358-11ea-8557-ef39a88028b2.png)


## Development roadmap

We aim to provide high level helper methods to load Argo data and meta-data from:
- [x] Ifremer erddap
- [x] local copy of the GDAC ftp folder
- [x] Index files
- [ ] any other usefull access point to Argo data ?

We also aim to provide high level helper methods to visualise and plot Argo data and meta-data:
- [ ] Map with trajectories
- [ ] Waterfall plots
- [ ] T/S diagram
- [ ] etc !


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


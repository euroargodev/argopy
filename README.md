# <img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/> Argo data python library

[![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest/?badge=latest)
![Github Action Status](https://github.com/euroargodev/argopy/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/euroargodev/argopy/branch/region-for-localftp/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)
[![Requirements Status](https://requires.io/github/euroargodev/argopy/requirements.svg?branch=master)](https://requires.io/github/euroargodev/argopy/requirements/?branch=master)
[![Gitter](https://badges.gitter.im/Argo-floats/argopy.svg)](https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![Profile count](https://img.shields.io/endpoint?label=Available%20Argo%20profile%20count%3A&style=social&url=https%3A%2F%2Fmap.argo-france.fr%2Fdata%2FARGOFULL.json)

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


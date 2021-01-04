|<img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="400"/>|``argopy`` is a python library that aims to ease Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators|
|:---------:|:-------|
|Documentation|[![JOSS](https://img.shields.io/badge/DOI-10.21105%2Fjoss.02425-brightgreen)](//dx.doi.org/10.21105/joss.02425) <br>[![Documentation](https://img.shields.io/static/v1?label=&message=Read%20the%20documentation&color=blue&logo=read-the-docs&logoColor=white)](https://argopy.readthedocs.io) <br>[![Gitter](https://badges.gitter.im/Argo-floats/argopy.svg)](https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)|
|Usage|![License](https://img.shields.io/github/license/euroargodev/argopy) [![Python version](https://img.shields.io/pypi/pyversions/argopy)](//pypi.org/project/argopy/) [![Requirements Status](https://requires.io/github/euroargodev/argopy/requirements.svg?branch=master)](https://requires.io/github/euroargodev/argopy/requirements/?branch=master)<br>[![pypi dwn](https://img.shields.io/pypi/dm/argopy?label=Pypi%20downloads)](//pypi.org/project/argopy/) [![conda dwn](https://img.shields.io/conda/dn/conda-forge/argopy?label=Conda%20downloads)](//anaconda.org/conda-forge/argopy)|
|Release|[![](https://img.shields.io/github/release-date/euroargodev/argopy)](//github.com/euroargodev/argopy/releases) <br>[![PyPI](https://img.shields.io/pypi/v/argopy)](//pypi.org/project/argopy/) [![Conda](https://anaconda.org/conda-forge/argopy/badges/version.svg)](//anaconda.org/conda-forge/argopy)|
|Development|![Github Action Status](https://github.com/euroargodev/argopy/workflows/tests/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)<br>[![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://www.tidyverse.org/lifecycle/#maturing)|
|Data resources|![Erddap status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_erddap.json) ![Argovis status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_argovis.json) <br>![Profile count](https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json) <br>[![Statuspage](https://img.shields.io/static/v1?label=&message=Check%20all%20monitors&color=blue&logo=statuspage&logoColor=white)](https://argopy.statuspage.io)|

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

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge)](https://binder.pangeo.io/v2/gh/euroargodev/argopy/master?urlpath=lab/tree/docs/tryit.ipynb)

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

Our next big steps:
- [ ] To provide Bio-geochemical variables

We aim to provide high level helper methods to load Argo data and meta-data from:
- [x] Ifremer erddap
- [x] local copy of the GDAC ftp folder
- [x] Index files (local and online)
- [x] Argovis
- [ ] Online GDAC ftp
- [ ] any other useful access point to Argo data ?

We also aim to provide high level helper methods to visualise and plot Argo data and meta-data:
- [ ] Map with trajectories
- [ ] Waterfall plots
- [ ] T/S diagram
- [ ] etc !


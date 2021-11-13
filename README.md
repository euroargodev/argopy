|<img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="400"/>|``argopy`` is a python library that aims to ease Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators|
|:---------:|:-------|
|Documentation|[![JOSS](https://img.shields.io/badge/DOI-10.21105%2Fjoss.02425-brightgreen)](//dx.doi.org/10.21105/joss.02425) <br>[![Documentation](https://img.shields.io/static/v1?label=&message=Read%20the%20documentation&color=blue&logo=read-the-docs&logoColor=white)](https://argopy.readthedocs.io) <br>[![Gitter](https://badges.gitter.im/Argo-floats/argopy.svg)](https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)|
|Usage|![License](https://img.shields.io/github/license/euroargodev/argopy) [![Python version](https://img.shields.io/pypi/pyversions/argopy)](//pypi.org/project/argopy/)<br>[![pypi dwn](https://img.shields.io/pypi/dm/argopy?label=Pypi%20downloads)](//pypi.org/project/argopy/) [![conda dwn](https://img.shields.io/conda/dn/conda-forge/argopy?label=Conda%20downloads)](//anaconda.org/conda-forge/argopy)|
|Release|[![](https://img.shields.io/github/release-date/euroargodev/argopy)](//github.com/euroargodev/argopy/releases) [![PyPI](https://img.shields.io/pypi/v/argopy)](//pypi.org/project/argopy/) [![Conda](https://anaconda.org/conda-forge/argopy/badges/version.svg)](//anaconda.org/conda-forge/argopy)|
|Development|![Github Action Status](https://github.com/euroargodev/argopy/workflows/tests/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)<br>[![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://www.tidyverse.org/lifecycle/#maturing)|
|Data resources|![Erddap status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_erddap.json) ![Argovis status](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/euroargodev/argopy-status/master/argopy_api_status_argovis.json) <br>![Profile count](https://img.shields.io/endpoint?label=Number%20of%20Argo%20profiles%3A&style=social&url=https%3A%2F%2Fapi.ifremer.fr%2Fargopy%2Fdata%2FARGO-FULL.json) <br>[![Statuspage](https://img.shields.io/static/v1?label=&message=Check%20all%20monitors&color=blue&logo=statuspage&logoColor=white)](https://argopy.statuspage.io)|

## Install


Install the last release with pip:
```bash
pip install argopy
```

But since this is a young library in active development, use direct install from this repo to benefit from the latest version:

```bash
pip install git+http://github.com/euroargodev/argopy.git@master
```

The ``argopy`` library is tested to work under most OS (Linux, Mac) and with python versions 3.7 and 3.8.

## Usage

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge)](https://binder.pangeo.io/v2/gh/euroargodev/argopy/master?urlpath=lab/tree/docs/tryit.ipynb)

### Fetching Argo Data

Import the data fetcher:
```python
from argopy import DataFetcher as ArgoDataFetcher
```
and then, set it up to request data for a **specific space/time domain**:
```python
argo_loader = ArgoDataFetcher().region([-85,-45,10.,20.,0,10.])
argo_loader = ArgoDataFetcher().region([-85,-45,10.,20.,0,1000.,'2012-01','2012-12'])
```
for **profiles of a given float**: 
```python
argo_loader = ArgoDataFetcher().profile(6902746, 34)
argo_loader = ArgoDataFetcher().profile(6902746, np.arange(12,45))
argo_loader = ArgoDataFetcher().profile(6902746, [1,12])
```
or for **one or a collection of floats**:
```python
argo_loader = ArgoDataFetcher().float(6902746)
argo_loader = ArgoDataFetcher().float([6902746, 6902747, 6902757, 6902766])
```

Once your fetcher is initialized you can trigger fetch/load data like this:
```python
ds = argo_loader.to_xarray()  # or:
ds = argo_loader.load().data
```
By default fetched data are returned in memory as [xarray.DataSet](http://xarray.pydata.org/en/stable/data-structures.html#dataset). 
From there, it is easy to convert it to other formats like a [Pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe):
```python
df = ArgoDataFetcher().profile(6902746, 34).load().data.to_dataframe()
```

or to export it to files:
```python
ds = ArgoDataFetcher().region([-85,-45,10.,20.,0,100.]).to_xarray()
ds.to_netcdf('my_selection.nc')
# or by profiles:
ds.argo.point2profile().to_netcdf('my_selection.nc')
```


### Fetching only Argo index
Argo index are returned as pandas dataframe. Index fetchers works similarly to data fetchers.

Load the Argo index fetcher:
```python
    from argopy import IndexFetcher as ArgoIndexFetcher
```
then, set it up to request index for a **specific space/time domain**:
```python
    index_loader = ArgoIndexFetcher().region([-85,-45,10.,20.])
    index_loader = ArgoIndexFetcher().region([-85,-45,10.,20.,'2012-01','2014-12'])
```
or for **one or a collection of floats**:
```python
    index_loader = ArgoIndexFetcher().float(6902746)
    index_loader = ArgoIndexFetcher().float([6902746, 6902747, 6902757, 6902766])   
```
Once your fetcher is initialized you can trigger fetch/load index like this:
```python
    df = index_loader.to_dataframe()  # or
    df = index_loader.load().index
```

Note that like the data fetcher, the index fetcher can use different sources, a local copy of the GDAC ftp for instance:
```python
    index_fetcher = ArgoIndexFetcher(src='localftp', path_ftp='/path/to/your/argo/ftp/', index_file='ar_index_global_prof.txt')
```

### Visualisation
For plottings methods, you'll need `matplotlib` and possibly `cartopy` and `seaborn` installed.
Argo Data and Index fetchers provide direct plotting methods, for instance:
```python    
    ArgoDataFetcher().float([6902745, 6902746]).plot('trajectory')    
```
![index_traj](https://github.com/euroargodev/argopy/raw/master/docs/_static/trajectory_sample.png)

See the [documentation page for more examples](https://argopy.readthedocs.io/en/latest/visualisation.html).

## Development roadmap

Our next big steps:
- [ ] To provide Bio-geochemical variables ([#22](https://github.com/euroargodev/argopy/issues/22), [#77](https://github.com/euroargodev/argopy/issues/77), [#81](https://github.com/euroargodev/argopy/issues/81))
- [ ] To develop expert methods related to Quality Control of the data with other python software like: 
  - [ ] [pyowc](https://github.com/euroargodev/argodmqc_owc): [#33](https://github.com/euroargodev/argodmqc_owc/issues/33), [#53](https://github.com/euroargodev/argodmqc_owc/issues/53)
  - [ ] [bgcArgoDMQC](https://github.com/ArgoCanada/bgcArgoDMQC): [#37](https://github.com/ArgoCanada/bgcArgoDMQC/issues/37)

We aim to provide high level helper methods to load Argo data and meta-data from:
- [x] Ifremer erddap
- [x] local copy of the GDAC ftp folder
- [x] Index files (local and online)
- [x] Argovis
- [ ] Online GDAC ftp

We also aim to provide high level helper methods to visualise and plot Argo data and meta-data:
- [x] Map with trajectories
- [x] Histograms for meta-data
- [ ] Waterfall plots
- [ ] T/S diagram
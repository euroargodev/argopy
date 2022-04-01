|<img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/><br>``argopy`` is a python library dedicated to Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators|
|:---------:|
|[![JOSS](https://img.shields.io/badge/DOI-10.21105%2Fjoss.02425-brightgreen)](//dx.doi.org/10.21105/joss.02425) ![CI](https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg) [![codecov](https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy) [![Documentation Status](https://img.shields.io/readthedocs/argopy?logo=readthedocs)](https://argopy.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/argopy)](//pypi.org/project/argopy/)|

### Documentation

The official documentation is hosted on ReadTheDocs.org: https://argopy.readthedocs.io

### Install

Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/argopy/) and on [Conda](https://anaconda.org/conda-forge/argopy).

```bash
# conda
conda install -c conda-forge argopy
````
```bash
# or PyPI
pip install argopy
````

``argopy`` is continuously tested to work under most OS (Linux, Mac, Windows) and with python versions 3.7 and 3.8.

### Usage

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Binder&message=Click+here+to+try+argopy+online+!&color=blue&style=for-the-badge)](https://mybinder.org/v2/gh/euroargodev/binder-sandbox/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Feuroargodev%252Fargopy%26urlpath%3Dlab%252Ftree%252Fargopy%252Fdocs%252Ftryit.ipynb%26branch%3Dmaster)

```python
# Import the main fetcher:
from argopy import DataFetcher as ArgoDataFetcher
```
```python
# Define what you want to fetch... 
# a region:
ArgoSet = ArgoDataFetcher().region([-85,-45,10.,20.,0,10.])
# floats:
ArgoSet = ArgoDataFetcher().float([6902746, 6902747, 6902757, 6902766])
# or specific profiles:
ArgoSet = ArgoDataFetcher().profile(6902746, 34)
```
```python
# then fetch and get data as xarray datasets:
ds = ArgoSet.load().data
# or
ds = ArgoSet.to_xarray()
```
```python
# you can even plot some information:
ArgoSet.plot('trajectory')    
```

They are many more usages and fine-tuning to allow you to access and manipulate Argo data:
- [filters at fetch time](https://argopy.readthedocs.io/en/latest/user_mode.html) (standard vs expert users, automatically select QC flags or data mode, ...)
- [select data sources](https://argopy.readthedocs.io/en/latest/data_sources.html) (erddap, ftp, local, ...)
- [manipulate data](https://argopy.readthedocs.io/en/latest/data_manipulation.html) (points, profiles, interpolations, binning, ...)
- [visualisation](https://argopy.readthedocs.io/en/latest/visualisation.html) (trajectories, topography, histograms, ...)
- [tools for Quality Control](https://argopy.readthedocs.io/en/latest/data_quality_control.html) (OWC, figures, ...)
- [improve performances](https://argopy.readthedocs.io/en/latest/performances.html) (caching, parallel data fetching)

Just check out [the documentation for more](https://argopy.readthedocs.io) ! 

## Development and contributions 

See our development roadmap here: https://github.com/euroargodev/argopy/milestone/3

Checkout [the contribution page](https://argopy.readthedocs.io/en/latest/contributing.html) if you want to get involved and help maintain or develop ``argopy``.

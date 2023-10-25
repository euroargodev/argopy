|<img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/><br>``argopy`` is a python library dedicated to Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators|
|:---------:|
|[![DOI][joss-badge]][joss-link] ![CI][ci-badge] [![codecov][cov-badge]][conda-link] [![Documentation][rtd-badge]][rtd-link] [![Pypi][pip-badge]][pip-link] [![Conda][conda-badge]][conda-link]|

[joss-badge]: https://img.shields.io/badge/DOI-10.21105%2Fjoss.02425-brightgreen
[joss-link]: https://dx.doi.org/10.21105/joss.02425
[ci-badge]: https://github.com/euroargodev/argopy/actions/workflows/pytests.yml/badge.svg
[cov-badge]: https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg
[cov-link]: https://codecov.io/gh/euroargodev/argopy
[rtd-badge]: https://img.shields.io/readthedocs/argopy?logo=readthedocs
[rtd-link]: https://argopy.readthedocs.io/en/latest/?badge=latest
[pip-badge]: https://img.shields.io/pypi/v/argopy
[pip-link]: https://pypi.org/project/argopy/
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/argopy?logo=anaconda
[conda-link]: https://anaconda.org/conda-forge/argopy

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

``argopy`` is continuously tested to work under most OS (Linux, Mac, Windows) and with python versions >= 3.8

### Usage

```python
# Import the main data fetcher:
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
- [select data sources](https://argopy.readthedocs.io/en/latest/data_sources.html) (erddap, ftp, local, argovis, ...)
- [manipulate data](https://argopy.readthedocs.io/en/latest/data_manipulation.html) (points, profiles, interpolations, binning, ...)
- [visualisation](https://argopy.readthedocs.io/en/latest/visualisation.html) (trajectories, topography, histograms, ...)
- [tools for Quality Control](https://argopy.readthedocs.io/en/latest/data_quality_control.html) (OWC, figures, ...)
- [access meta-data and other Argo-related datasets](https://argopy.readthedocs.io/en/latest/metadata_fetching.html) (index, reference tables, deployment plans, topography, ...)
- [improve performances](https://argopy.readthedocs.io/en/latest/performances.html) (caching, parallel data fetching)

Just check out [the documentation for more](https://argopy.readthedocs.io) ! 

## Development and contributions 

See our development roadmap here: https://github.com/euroargodev/argopy/milestone/3

Checkout [the contribution page](https://argopy.readthedocs.io/en/latest/contributing.html) if you want to get involved and help maintain or develop ``argopy``.

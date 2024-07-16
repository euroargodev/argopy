| <img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/><br>``argopy`` is a python library dedicated to Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                         [![DOI][joss-badge]][joss-link] [![Documentation][rtd-badge]][rtd-link] [![Pypi][pip-badge]][pip-link] [![Conda][conda-badge]][conda-link]                                                          |
|                                                                                                                 [![codecov][cov-badge]][conda-link]  ![CI][ci-badge] [![CI Energy][ci-energy-badge]][ci-energy-link]                                                                                                                  |
| [![Open-SSF][ossf-badge]][ossf-link] |

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
[ossf-badge]: https://www.bestpractices.dev/projects/5939/badge
[ossf-link]: https://www.bestpractices.dev/projects/5939


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
from argopy import DataFetcher
```
```python
# Define what you want to fetch... 
# a region:
ArgoSet = DataFetcher().region([-85,-45,10.,20.,0,10.])
# floats:
ArgoSet = DataFetcher().float([6902746, 6902747, 6902757, 6902766])
# or specific profiles:
ArgoSet = DataFetcher().profile(6902746, 34)
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

## 🌿 Energy impact of **argopy** development

[ci-energy-badge]: https://api.green-coding.berlin/v1/ci/badge/get?repo=euroargodev/argopy&branch=master&workflow=22344160
[ci-energy-link]: https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=22344160

[ci-energy-badge-upstream]: https://api.green-coding.berlin/v1/ci/badge/get?repo=euroargodev/argopy&branch=master&workflow=25052179
[ci-energy-link-upstream]: https://metrics.green-coding.io/ci.html?repo=euroargodev/argopy&branch=master&workflow=25052179

The **argopy** team is concerned about the environmental impact of your favorite software development. Starting June 1st 2024, we're experimenting with the [Green Metrics Tools](https://metrics.green-coding.io) from [Green Coding](https://www.green-coding.io/) to get an estimate of the energy used and CO2eq emitted by our development activities on Github infrastructure. Results:

| Activity                         | Green Coding tool                                                                                                                           |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| All                              | [Dashboard](https://metrics.green-coding.io/carbondb-lists.html?project_uuid=a5c7557d-f668-482b-b740-b87d0bbf5b6d)                          |
| CI tests running on each commit  | [![CI Energy][ci-energy-badge]][ci-energy-link] |
| Upstream CI tests, running daily | [![CI Energy][ci-energy-badge-upstream]][ci-energy-link-upstream]|



## Development and contributions 

See our development roadmap here: https://github.com/euroargodev/argopy/milestone/3

Checkout [the contribution page](https://argopy.readthedocs.io/en/latest/contributing.html) if you want to get involved and help maintain or develop ``argopy``.

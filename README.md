# <img src="https://raw.githubusercontent.com/euroargodev/argopy/master/docs/_static/argopy_logo_long.png" alt="argopy logo" width="200"/> Argo data python library

[![Documentation Status](https://readthedocs.org/projects/argopy/badge/?version=latest)](https://argopy.readthedocs.io/en/latest/?badge=latest)
![Github Action Status](https://github.com/euroargodev/argopy/workflows/build/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/euroargodev/argopy/branch/master/graph/badge.svg)](https://codecov.io/gh/euroargodev/argopy)
[![Requirements Status](https://requires.io/github/euroargodev/argopy/requirements.svg?branch=master)](https://requires.io/github/euroargodev/argopy/requirements/?branch=master)
[![Gitter](https://badges.gitter.im/Argo-floats/argopy.svg)](https://gitter.im/Argo-floats/argopy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

![Profile count](https://img.shields.io/endpoint?style=social&url=https%3A%2F%2Fmap.argo-france.fr%2Fdata%2FARGOFULL.json&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABYWlDQ1BrQ0dDb2xvclNwYWNlRGlzcGxheVAzAAAokWNgYFJJLCjIYWFgYMjNKykKcndSiIiMUmB/yMAOhLwMYgwKicnFBY4BAT5AJQwwGhV8u8bACKIv64LMOiU1tUm1XsDXYqbw1YuvRJsw1aMArpTU4mQg/QeIU5MLikoYGBhTgGzl8pICELsDyBYpAjoKyJ4DYqdD2BtA7CQI+whYTUiQM5B9A8hWSM5IBJrB+API1klCEk9HYkPtBQFul8zigpzESoUAYwKuJQOUpFaUgGjn/ILKosz0jBIFR2AopSp45iXr6SgYGRiaMzCAwhyi+nMgOCwZxc4gxJrvMzDY7v////9uhJjXfgaGjUCdXDsRYhoWDAyC3AwMJ3YWJBYlgoWYgZgpLY2B4dNyBgbeSAYG4QtAPdHFacZGYHlGHicGBtZ7//9/VmNgYJ/MwPB3wv//vxf9//93MVDzHQaGA3kAFSFl7jXH0fsAAAAJcEhZcwAACxMAAAsTAQCanBgAAAFZaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA1LjQuMCI+CiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+CiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIj4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgPC9yZGY6RGVzY3JpcHRpb24+CiAgIDwvcmRmOlJERj4KPC94OnhtcG1ldGE+CkzCJ1kAAAvxSURBVFgJjVcLcFTVGf7u3Xt3N5tsnptk8yIhxDwhIgoNQRrxBT6qfcWCTDs6tmIfDG3tWNtSq1a0UxzGZw1MHa1PFOnLIlKVkAdIkEBICNmEvGCT7Cub3WTfe+/e0/8soY11OtMzc+aee+45//e///8K+N9DaG1t1a1bt06dP1Lw2r59TfVVVWtMBkO9LOqWSLKUyr8lEolgNB6/EI3FevpHz3du/npzG207+DfGmCQIQoIv+fv/NZqbm3ULDla3tre/5JuevhgNBBhTE8zt8bC+8THWNTqcnHztoj2WSDB+xu/1Xmzr7GwhGrWX6bz77rsLaV7ehvDv1fzi0UcflWhyqQ0HDh36xarlyx+yZGWnDE5N4kQooJ6WRXhNKaJmMAqCdIkmAUOMxlhOKKwtVxNYZc6QqgsKMT0zE+06fXrX7Rs27CB64QW059HweQZ2794tb9myRaGvZX39/W8tra1dbRsbZe8E5xKD1nzRTMAZOgkyKVNg2rxS+YsAJohQSBy/qmJ2bk6rdrm1jZnZuprF5UK/zfbZ0pqaTUR3ZAFGkol/a2ABd7UTk5MfFhUUlLzW1aXsz8uTrAa9YAiFIGdlQdTpuM2ToJyCQMCcE6ZpSWnI4NDJMoKigBGnizV7POo9X2qQJ5xOR0lx8QY63LsA65IGuM337dvHHaXsot3eUWCxFD92vEs5X10tl9Lm6/ffj5LTPWg+9CHs+VZIHJJYl0QdYrEYdASqNxqhMQaNGEnQHmcr1WzGxbiCRefOKr++arns8vudJUVFa+nTMPeJu+66K8E1wCf3UCOp/fDS2prVv/6kTRmtrJLL01Jx6G9/x7oTR5MSFmz+Duy1S+HuPQMzER+ZnMSyujrESOLuY8egDwZRUFYGS00tdMRMX9dx6Hw+mBtWo2ZmWt2xtlEaOD98sra6uonwwhxb5KFGC7x/4MDD3Oavd3yq9BQvlgtJBqdrGlkfH8JtP34Qs9m5UKa9kBMMZ/a9hzO3r0TzyDmsVBW8eN99+O7kKNbY+pG55yXUawzP7XwaNx1vx83OCYxs2wpbeo706vGTSk1V1TUHDh7czjEvY/N1lcflCtlGRrXGYz3atuFJ9sNRJ7vlzf2M4oj96s29XEPs92+9w356YZph/a2s5cU97GNPmN345NPskS0PsM4oY03P72Yv7G5huzqPs2vo/HGHmz02OM4yab3neDdr7B7Qzg4NadMed4ToJUOUmwqHW9u2WnItpj+N2BPlmdlCLBpHPBjGwT178J0dv0Udhd6tN1yPOc8UcoY/Aw79E41NJcidexw4+jMUXLkYY6feRtvWLTBaS3HGNYdlN67CaVXEb46fwoZlZbBmZiI/LV14fWwqkZOdbWzr6NjGsTkD+SuuXHr70EU7PtGZRXNcJQeTMHi0A9uLDGi89/sYXnkzJktWwBKbxm2pB+jKJpRYA6gvv4CHfrQZz7zYgdmONqxecjVK0kMoT1zAKx9n4sAT23HnG8/jgedewaScBmMoiqOGDHFg3I762trbiFCh9NY7711r1OtLOy441cyUTEmLJzATS+CGikrcdMMzeHXYi6bgc/jbw2kQjDcg1fAZBgfvoOcw4nMCGldl4ciHX8PQiIbAvjCqSyWsXfYXNBxYB0d0Ba6oW4IRKQ0fuHwoosQ1ozeJHXanem9hftHb+/dfJ9VUVqwxmFJxZCYCS34OIjEFBrK4I9uKbWNhPCb+AbdcOw0mipTX91KYMWSbR6HEYySACbv3jGPK82eMjsew45EtYOp5pJiycfP1NrzfZcQTrnKIghcWuh8i2hZJRpsvivsNRtRUVKyVTCbD1W7ybpuqE+tUDTElgRSdgG53GFuj72BD0xTCMStVnEAyYAWK2jiFmCjK5FpBfPPrN2IuvATWAjNyMhzoPPIn2C8UYWw8iO7DtC6JYfWd30AwGiWGBRJOQ68qiU6PB1TUrqSCpi9z+2bh00SBkfpBuXxGM2LldC/uWPMpYupiMCVEAUvuQpqhnEeEiB81TkwYsKjgNNR4LwaGLuLlQx/j1betsA2dSJ4B8uj5MDIr61FaXo54LAo9ud0sEwW3bw5mWS6XZFlKD1ECCVH50RQVItG3zZL0ee1ITc1GOEJZjTaZxhPl/KAzPAPHYhqOtLux/+/n8fIb4/Qxm6YTJcUyZEnEqF+HFZTmrps9jJ5wCVJIcwmoCBNWiCItNzXdKJFKk0KFSPWMIiDAdFgWmERN9RDica56hex+CfhyymTkB0ajhL8cHMO3f3As+bFskYFkC2IuICIR15DMbn4HGu4tx90Ng+gaclEtyYFeUxEkTVOfkLwnxVUlkGrUZwUjVATNGpxxEXfq7EhPUaCQhGAkOUemwe8IVGT4QiGHGh7xJ/dXLEuDfTICz8wCLZGsfFjJsYssIVSedeBczAKjLkHSK0gz6hFV4lFJiSsX8rJzFmVHhlhCZYIrrKDQ4CHP1SFOJrnUzBClefBoRIWkE6Ez6CDLlzg71RdMgn1vcymZiqGsJI3uCwiE4jDQGUnUsEjnRUckgTyjhoxIlOVlZQiBOd+oFI5Euitzc9Y2CooWjKoiFTKkGMNJwTVySDHZTV2SXq8X4abo6BuYxaavLkZRXkoSeO8fvoSy4jTk56bgVJ8XDdfkIUyGzrMYMRsguyvTSBVCFOKkfmhoYIpmzc8VXW5Xr3hucPBYPBrBrSVm2PxhpJIECllDo8aC2i96asnJ6Bkn6YtyTTjV00tzAlZLOnb+ajmaVhVAL+sQCipwuincVIYgrSMhBSa9RPcT5E8CjFTIOMbtxWamUh6xnRvsEDdv3NgZicbs6+rLpGG3T0slJ3QH0siRIqQF8s8EMUINiEYzQUwIWhgVFV/Bs69nwO04jCuKAzh56gzO9A4jGJjFzPQUjHIMZiPVk0gAshglWiE4gmkwkzONePzadfWlcigcmdy4sfkI7y0cp86c/eu6prVbt2V3a69GVHEono9QKAbZQNLzECB7JsOfnjElBv+cgpqrr8fF2GronNPIS/fBF3BjxC7C4Tfio6MGxFUZkagOkqTBS2YboOzoFxU8kK5qtVXlYltb5weEPTnv36igEtnj8c6aap7vRrUlQ3ilbieuKJMRI9UJFKkaacNkkjDlDOJ7vw3h2rVNICmQkZULUTKiuDAfQSo2JgpP31yUcodAfhBBTnYmPj09AEflreiwz7CzP1yOgnxLLMdiuZoYOMcbEq6F4a4T3U9XV1UIL69JU22OBDrGm5CIjpMDUUQnNKRQSXY4I3julXbkWhfDarXio0MH8dSOx7HjsV/itT8+SwWKCpLfgwyTBlH1IyddQtQ/Dl1eFToccbSsTlXraquErpMnd3Fwjs01wCfPCsbevr7WZUuXNjz0uzeUnX2y3LrhLSyv8MHp1eODdjse3DlEx/57mGmDR4MbyLwKv9j2LYyNjcBaWIqQdwKWwkV4auIK/KQsouza/m15wDZ4srbmPy1Z0gQLm1L7xER7Yb6lZNuT+9UXTjikj+58l4oGcHFKgWOanIpS7GIKuR5bCFOuAIWeHivrssg3BLinZ1GQq6LYWgzf7CRau3PwRP89eHBDlvq7n39DcnpmvtCU8oYEvCPmrTItx6l1Xu9wee3PP3K39FJzXXz9zkY2NzuDZZVmIp4Dc6oe5UUpuGVNFhquzMQdTQWoLE2DXkdmSklFQV4FXN4I+YEFJ123sRe/u0R5evtmyen2OqgjXk8YwxyLd8Qc+3OD/zTMb5T19/d/Svma9Q+MaJu//xT/WeEXuKlYYW42u6q2hNaFbHlNKWuoX0Tr9OS3+TOJ9ffuUnrPDlOnTjT6B3h5XEITCzD46xfHvCb4B+M//vHB414vdZ40bIPn2Z7dLcr1TY2XmeElauFMXPflBqWlpUWxDQ7zK2zG64m8f+Dgk3TOxAkuoM1f//f4ws9pa2uL3++zx+PUitBwOp2st7eXdXZ2Jmdv75nkHv+mKArz+WYm2tvb9xBC3WWU//vn9PIFev7373nh3r17v1xVVbXWZDLV6/X6xTS5ZAIxFqY5Hg6He2w229FNmzYdof0pmrz3l+gX/7L5+Nbnxr8AcUAP6J0ztkYAAAAASUVORK5CYII=)

``argopy`` is a python library that aims to ease Argo data access, visualisation and manipulation for regular users as well as Argo experts and operators.

Several python packages exist: we are currently in the process of analysing how to merge these libraries toward a single powerfull tool.  
[List your tool here !](https://github.com/euroargodev/argopy/issues/3)

Click here to [![badge](https://img.shields.io/badge/launch-Pangeo%20binder-579ACA.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAABZCAMAAABi1XidAAAB8lBMVEX///9XmsrmZYH1olJXmsr1olJXmsrmZYH1olJXmsr1olJXmsrmZYH1olL1olJXmsr1olJXmsrmZYH1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olJXmsrmZYH1olL1olL0nFf1olJXmsrmZYH1olJXmsq8dZb1olJXmsrmZYH1olJXmspXmspXmsr1olL1olJXmsrmZYH1olJXmsr1olL1olJXmsrmZYH1olL1olLeaIVXmsrmZYH1olL1olL1olJXmsrmZYH1olLna31Xmsr1olJXmsr1olJXmsrmZYH1olLqoVr1olJXmsr1olJXmsrmZYH1olL1olKkfaPobXvviGabgadXmsqThKuofKHmZ4Dobnr1olJXmsr1olJXmspXmsr1olJXmsrfZ4TuhWn1olL1olJXmsqBi7X1olJXmspZmslbmMhbmsdemsVfl8ZgmsNim8Jpk8F0m7R4m7F5nLB6jbh7jbiDirOEibOGnKaMhq+PnaCVg6qWg6qegKaff6WhnpKofKGtnomxeZy3noG6dZi+n3vCcpPDcpPGn3bLb4/Mb47UbIrVa4rYoGjdaIbeaIXhoWHmZYHobXvpcHjqdHXreHLroVrsfG/uhGnuh2bwj2Hxk17yl1vzmljzm1j0nlX1olL3AJXWAAAAbXRSTlMAEBAQHx8gICAuLjAwMDw9PUBAQEpQUFBXV1hgYGBkcHBwcXl8gICAgoiIkJCQlJicnJ2goKCmqK+wsLC4usDAwMjP0NDQ1NbW3Nzg4ODi5+3v8PDw8/T09PX29vb39/f5+fr7+/z8/Pz9/v7+zczCxgAABC5JREFUeAHN1ul3k0UUBvCb1CTVpmpaitAGSLSpSuKCLWpbTKNJFGlcSMAFF63iUmRccNG6gLbuxkXU66JAUef/9LSpmXnyLr3T5AO/rzl5zj137p136BISy44fKJXuGN/d19PUfYeO67Znqtf2KH33Id1psXoFdW30sPZ1sMvs2D060AHqws4FHeJojLZqnw53cmfvg+XR8mC0OEjuxrXEkX5ydeVJLVIlV0e10PXk5k7dYeHu7Cj1j+49uKg7uLU61tGLw1lq27ugQYlclHC4bgv7VQ+TAyj5Zc/UjsPvs1sd5cWryWObtvWT2EPa4rtnWW3JkpjggEpbOsPr7F7EyNewtpBIslA7p43HCsnwooXTEc3UmPmCNn5lrqTJxy6nRmcavGZVt/3Da2pD5NHvsOHJCrdc1G2r3DITpU7yic7w/7Rxnjc0kt5GC4djiv2Sz3Fb2iEZg41/ddsFDoyuYrIkmFehz0HR2thPgQqMyQYb2OtB0WxsZ3BeG3+wpRb1vzl2UYBog8FfGhttFKjtAclnZYrRo9ryG9uG/FZQU4AEg8ZE9LjGMzTmqKXPLnlWVnIlQQTvxJf8ip7VgjZjyVPrjw1te5otM7RmP7xm+sK2Gv9I8Gi++BRbEkR9EBw8zRUcKxwp73xkaLiqQb+kGduJTNHG72zcW9LoJgqQxpP3/Tj//c3yB0tqzaml05/+orHLksVO+95kX7/7qgJvnjlrfr2Ggsyx0eoy9uPzN5SPd86aXggOsEKW2Prz7du3VID3/tzs/sSRs2w7ovVHKtjrX2pd7ZMlTxAYfBAL9jiDwfLkq55Tm7ifhMlTGPyCAs7RFRhn47JnlcB9RM5T97ASuZXIcVNuUDIndpDbdsfrqsOppeXl5Y+XVKdjFCTh+zGaVuj0d9zy05PPK3QzBamxdwtTCrzyg/2Rvf2EstUjordGwa/kx9mSJLr8mLLtCW8HHGJc2R5hS219IiF6PnTusOqcMl57gm0Z8kanKMAQg0qSyuZfn7zItsbGyO9QlnxY0eCuD1XL2ys/MsrQhltE7Ug0uFOzufJFE2PxBo/YAx8XPPdDwWN0MrDRYIZF0mSMKCNHgaIVFoBbNoLJ7tEQDKxGF0kcLQimojCZopv0OkNOyWCCg9XMVAi7ARJzQdM2QUh0gmBozjc3Skg6dSBRqDGYSUOu66Zg+I2fNZs/M3/f/Grl/XnyF1Gw3VKCez0PN5IUfFLqvgUN4C0qNqYs5YhPL+aVZYDE4IpUk57oSFnJm4FyCqqOE0jhY2SMyLFoo56zyo6becOS5UVDdj7Vih0zp+tcMhwRpBeLyqtIjlJKAIZSbI8SGSF3k0pA3mR5tHuwPFoa7N7reoq2bqCsAk1HqCu5uvI1n6JuRXI+S1Mco54YmYTwcn6Aeic+kssXi8XpXC4V3t7/ADuTNKaQJdScAAAAAElFTkSuQmCC)](https://binder.pangeo.io/v2/gh/euroargodev/argopy/master?urlpath=lab/tree/docs/tryit.ipynb) and play with ``argopy`` before you even install it (thanks [Pangeo](pangeo.io)).

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


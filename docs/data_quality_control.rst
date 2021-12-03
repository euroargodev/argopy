.. _data_qc:

Data quality control
====================

**argopy** comes with handy methods to help you quality control measurements. This section is probably intended for `expert` users.

Most of these methods are available through the :class:`xarray.Dataset` accessor namespace ``argo``. This means that if your dataset is `ds`, then you can use `ds.argo` to access more **argopy** functionalities.

Let's start with import and set-up:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher as ArgoDataFetcher

Topography
----------
.. currentmodule:: argopy

For some QC of trajectories, it can be useful to easily get access to the topography. This can be done with the **argopy** utility :class:`TopoFetcher`:

.. ipython:: python
    :okwarning:

    from argopy import TopoFetcher
    box = [-65, -55, 10, 20]
    ds = TopoFetcher(box, cache=True).to_xarray()

.. image:: _static/topography_sample.png


Combined with the fetcher property ``domain``, it now becomes easy to superimpose float trajectory with topography:

.. ipython:: python
    :okwarning:

    fetcher = ArgoDataFetcher().float(2901623)
    ds = TopoFetcher(fetcher.domain[0:4], cache=True).to_xarray()

.. code-block:: python

    fig, ax = fetcher.plot('trajectory', figsize=(10, 10))
    ds['elevation'].plot.contourf(levels=np.arange(-6000,0,200), ax=ax, add_colorbar=False)

.. image:: _static/trajectory_topography_sample.png


.. note::
    The :class:`TopoFetcher` can return a lower resolution topography with the ``stride`` option. See the :class:`argopy.TopoFetcher` full documentation for all the details.


Altimetry
---------
.. currentmodule:: argopy

Satellite altimeter measurements can be used to check the quality of the Argo profiling floats time series. The method compares collocated sea level anomalies from altimeter measurements and dynamic height anomalies calculated from Argo temperature and salinity profiles for each Argo float time series [Guinehut2008]_. This method is performed routinely by CLS and results are made available online.


**argopy** provides a simple access to this QC analysis with an option to the data and index fetchers :meth:`DataFetcher.plot` methods that will insert the CLS Satellite Altimeter report figure on a notebook cell.

.. code-block:: python

    fetcher = ArgoDataFetcher().float(6902745)
    fetcher.plot('qc_altimetry', embed='list')

.. image:: https://data-argo.ifremer.fr/etc/argo-ast9-item13-AltimeterComparison/figures/6902745.png

See all details about this method here: :meth:`argopy.plotters.open_sat_altim_report`


.. rubric:: References

.. [Guinehut2008] Guinehut, S., Coatanoan, C., Dhomps, A., Le Traon, P., & Larnicol, G. (2009). On the Use of Satellite Altimeter Data in Argo Quality Control, Journal of Atmospheric and Oceanic Technology, 26(2), 395-402. `10.1175/2008JTECHO648.1 <https://doi.org/10.1175/2008JTECHO648.1>`_

.. currentmodule:: argopy

Altimetry
---------

Satellite altimeter measurements can be used to check the quality of the Argo profiling floats time series. The method compares collocated sea level anomalies from altimeter measurements and dynamic height anomalies calculated from Argo temperature and salinity profiles for each Argo float time series [Guinehut2008]_.

**argopy** does not provide a method to compute these anomalies but since this method is performed routinely by CLS and
that results are made available online, **argopy** provides a simple access to these results. This can be done using an option to the data fetcher :meth:`DataFetcher.plot` method that will insert the CLS Satellite Altimeter report figure on a notebook cell.

.. code-block:: python

    from argopy import DataFetcher

    fetcher = DataFetcher().float(6902745)
    fetcher.plot('qc_altimetry', embed='list')

.. image:: https://data-argo.ifremer.fr/etc/argo-ast9-item13-AltimeterComparison/figures/6902745.png

See all details about this method in here: :meth:`argopy.plot.open_sat_altim_report`


.. rubric:: References

.. [Guinehut2008] Guinehut, S., Coatanoan, C., Dhomps, A., Le Traon, P., & Larnicol, G. (2009). On the Use of Satellite Altimeter Data in Argo Quality Control, Journal of Atmospheric and Oceanic Technology, 26(2), 395-402. `10.1175/2008JTECHO648.1 <https://doi.org/10.1175/2008JTECHO648.1>`_

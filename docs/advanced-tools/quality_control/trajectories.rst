.. currentmodule:: argopy
.. _qc_traj:

Trajectories
------------

Topography
^^^^^^^^^^

For some QC of trajectories, it can be useful to easily get access to the topography. This can be done with the **argopy** utility :class:`TopoFetcher`:

.. code-block:: python

    from argopy import TopoFetcher
    box = [-65, -55, 10, 20]
    ds = TopoFetcher(box, cache=True).to_xarray()

.. image:: ../../_static/topography_sample.png


Combined with the fetcher property ``domain``, it now becomes easy to superimpose float trajectory with topography:

.. code-block:: python

    from argopy import DataFetcher

    fetcher = DataFetcher().float(2901623)
    ds = TopoFetcher(fetcher.domain[0:4], cache=True).to_xarray()

.. code-block:: python

    fig, ax = fetcher.plot('trajectory', figsize=(10, 10))
    ds['elevation'].plot.contourf(levels=np.arange(-6000,0,100), ax=ax, add_colorbar=False)

.. image:: ../../_static/trajectory_topography_sample.png


.. note::
    The :class:`TopoFetcher` can return a lower resolution topography with the ``stride`` option. See the :class:`argopy.TopoFetcher` full documentation for all the details.


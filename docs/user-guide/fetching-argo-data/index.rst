.. currentmodule:: argopy
.. _data_fetching:

Fetching Argo data
==================

To fetch (i.e. access, download, format) Argo data, **argopy** provides the :class:`DataFetcher` class. In this section of the documentation, we explain how to use it.

Several :class:`DataFetcher` arguments exist to help you select the :ref:`dataset <data-set>`, the :ref:`data source <data-sources>` and the :ref:`user mode <user-mode>` the most suited for your applications; and also to improve :ref:`performances <performances>`.

You define the selection of data you want to fetch with one of the :class:`DataFetcher` methods: :ref:`region <data-selection-region>`, :ref:`float <data-selection-float>` or :ref:`profile <data-selection-profile>`.

These methods and arguments are all explained in the following sections:

* :doc:`data_selection`
* :doc:`data_sources`
* :doc:`data_set`
* :doc:`user_mode`

.. toctree::
    :maxdepth: 3
    :hidden:
    :caption: Fetching Argo data

    data_selection
    data_sources
    data_set
    user_mode


In a nutshell
*************

2 lines to download Argo data: import and fetch !

.. ipython:: python
    :okwarning:

    import argopy
    ds = argopy.DataFetcher().region([-75, -45, 20, 30, 0, 10, '2011-01', '2011-06']).load().data

.. ipython:: python
    :okwarning:

    ds

Workflow explained
******************

Let's explain what happened in the single line Argo data fetching above.

.. tabs::

    .. tab:: 1 we create a DataFetcher

        Import **argopy** and create a instance of :class:`DataFetcher`:

        .. ipython:: python
            :okwarning:

            import argopy
            f = argopy.DataFetcher()
            f

        By default, **argopy** will load the ``phy`` :ref:`dataset <data-set>`, in ``standard`` :ref:`user mode <user-mode>` from the ``erddap`` :ref:`data source <data-sources>`.


    .. tab:: 2 we select data

        Once you have a :class:`DataFetcher`, you must select data. As an example, here is a space/time data selection:

        .. ipython:: python
            :okwarning:

            f = f.region([-75, -45, 20, 30, 0, 10, '2011-01', '2011-06'])
            f

        See :ref:`all data selector methods here <data-selection>`.

    .. tab:: 3 then we fetch data

        Once you defined a data selection, data fetching will be triggered if you access one of the :class:`DataFetcher` properties:

        - ``data``, this is a :class:`xarray.Dataset` with all Argo data in the selection,
        - ``index``, this is a :class:`pandas.DataFrame` with a list of profiles in the selection.

        .. ipython:: python
            :okwarning:

            f.data

        .. ipython:: python
            :okwarning:

            f.index

        These fetcher properties call on the :meth:`DataFetcher.load` method, which in turn, calls either :meth:`DataFetcher.to_xarray` or :meth:`DataFetcher.to_index` if data structures are not already in memory.


        Note that the :meth:`DataFetcher.to_xarray` and :meth:`DataFetcher.to_index` will force data download on every call, while the :meth:`DataFetcher.load` method will keep data in memory in the :attr:`DataFetcher.data` and :attr:`DataFetcher.index` properties.

    .. tab:: 4 Viz data

        If you wish to quickly look at your data selection, you can call on the :meth:`DataFetcher.plot`.

        .. code-block:: python

            f.plot('trajectory', add_legend=False)

        .. image:: ../../_static/trajectory_sample_region.png

        If you selected data for a float, the :meth:`DataFetcher.dashboard` method can also be used.

        See the :ref:`data-viz` section for more details on **argopy** data visualisation tools.

        .. hint::

            The :attr:`DataFetcher.domain` property will also give you the space/time domain covered by your data selection.

            .. ipython:: python
                :okwarning:

                f.domain  # [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, date_min, date_max]



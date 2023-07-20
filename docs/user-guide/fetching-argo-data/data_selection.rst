.. currentmodule:: argopy
.. _data-selection:

Data selection
==============

To access Argo data with a :class:`DataFetcher`, you need to define how to select your data of interest.

**argopy** provides 3 different data selection methods:

- :ref:`data-selection-region`,
- :ref:`data-selection-float`,
- :ref:`data-selection-profile`.

To show how these methods (i.e. *access points*) work, let's first create a :class:`DataFetcher`:

.. ipython:: python
    :okwarning:

    import argopy
    f = argopy.DataFetcher()
    f

By default, **argopy** will load the ``phy`` dataset (:ref:`see here for details <data-set>`), in ``standard`` user mode (:ref:`see here for details <user-mode>`) from the ``erddap`` data source (:ref:`see here for details <data-sources>`).

The standard :class:`DataFetcher` print indicates all available access points, and here, that none is selected yet.

.. _data-selection-region:

🗺 For a space/time domain
--------------------------

Use the fetcher access point :meth:`argopy.DataFetcher.region` to select data for a *rectangular* space/time domain. For instance, to retrieve data from 75W to 45W, 20N to 30N, 0db to 10db and from January to May 2011:

.. ipython:: python
    :okwarning:

    f = f.region([-75, -45, 20, 30, 0, 10, '2011-01-01', '2011-06'])
    f

You can now see that the standard :class:`DataFetcher` print has been updated with information for the data selection.

.. note::

    - The constraint on time is not mandatory: if not specified, the fetcher will return all data available in this region.
    - The last time bound is exclusive: that's why here we specify June to retrieve data collected in May.

.. _data-selection-float:

🤖 For one or more floats
-------------------------

If you know the Argo float unique identifier number called a `WMO number <https://www.wmo.int/pages/prog/amp/mmop/wmo-number-rules.html>`_ you can use the fetcher access point :meth:`DataFetcher.float` to specify one or more float WMO platform numbers to select.

For instance, to select data for float WMO *6902746*:

.. ipython:: python
    :okwarning:

    f = f.float(6902746)
    f

To fetch data for a collection of floats, input them in a list:

.. ipython:: python
    :okwarning:

    f = f.float([6902746, 6902755])
    f

.. _data-selection-profile:

⚓ For one or more profiles
---------------------------

Use the fetcher access point :meth:`argopy.DataFetcher.profile` to specify the float WMO platform number and the profile cycle number(s) to retrieve profiles for.

For instance, to retrieve data for the 12th profile of float WMO 6902755:

.. ipython:: python
    :okwarning:

    f.profile(6902755, 12).data

To fetch data for more than one profile, input them in a list:

.. ipython:: python
    :okwarning:

    f.profile(6902755, [3, 12]).data


.. note::

    You can chain data selection and fetching in a single command line:

    .. code-block:: python

        f = argopy.DataFetcher().region([-75, -45, 20, 30, 0, 10, '2011-01-01', '2011-06']).load()
        f.data


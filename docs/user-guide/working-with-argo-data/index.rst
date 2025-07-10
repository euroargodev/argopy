Working with Argo data
======================

**argopy** not only get you easy access to Argo data, it also aims to help you work with it.

In the following documentation sections, you will see how to:

- :ref:`manipulate Argo data <Manipulating data>` from a :class:`xarray.Dataset` with ``argo`` accessor methods,
- :ref:`compute new data <Computing new data>` from an Argo :class:`xarray.Dataset` with ``argo`` accessor methods,
- :ref:`save Argo data <Saving data>` from a :class:`xarray.Dataset` with ``argo`` accessor methods,
- :ref:`visualize Argo data <data-viz>`, whether it is a :class:`xarray.Dataset` or :class:`pandas.DataFrame` profile index,

You can also refer to the documentation on :ref:`expert-tools`.

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Working with Argo data

    data_manipulation
    data_computation
    data_saving
    visualisation


.. currentmodule:: argopy

.. admonition:: About **argopy** data model

    By default **argopy** will provide users with a :class:`xarray.Dataset` or :class:`pandas.DataFrame`.

    For your own analysis, you may prefer to switch from one to the other. This is all built in **argopy**, with the :meth:`DataFetcher.to_dataframe` and :meth:`DataFetcher.to_xarray` methods.

    .. ipython:: python
        :okwarning:

        from argopy import DataFetcher
        DataFetcher().profile(6902746, 34).to_dataframe()


    Note that internally, **argopy** also work with :class:`pyarrow.Table`.

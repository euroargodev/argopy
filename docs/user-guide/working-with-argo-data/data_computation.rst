Computing new data
==================

Once you fetched Argo data, **argopy** comes with a handy :class:`xarray.Dataset` accessor ``argo`` to perform new data computation. This means that if your dataset is named ``ds``, then you can use ``ds.argo`` to access more **argopy** functions. The full list is available in the API documentation page :ref:`Dataset.argo (xarray accessor)`.

In this section, we present how **argopy** can help compute new, complementary, data from Argo measurements and parameters.

.. contents::
   :local:

.. _complement-teos10:

TEOS-10 variables
-----------------

.. currentmodule:: xarray

You can compute additional ocean variables from `TEOS-10 <http://teos-10.org/>`_. The default list of variables is: 'SA', 'CT', 'SIG0', 'N2', 'PV', 'PTEMP' ('SOUND_SPEED', 'CNDC' are optional). `Simply raise an issue to add a new one <https://github.com/euroargodev/argopy/issues/new/choose>`_.

This can be done using the :meth:`Dataset.argo.teos10` method and indicating the list of variables you want to compute:

.. ipython:: python
    :okwarning:

    from argopy import DataFetcher

    ds = DataFetcher().float(2901623).to_xarray()
    ds.argo.teos10(['SA', 'CT', 'PV'])

.. ipython:: python
    :okwarning:

    ds['SA']

.. _complement-canyon-med:

Nutrient and carbonate system variables
---------------------------------------

.. currentmodule:: xarray

For BGC, it may be possible to complement a dataset with predictions of the water-column nutrient concentrations and carbonate system variables.

This is currently possible for data located in the Mediterranean Sea using the the CANYON-MED model. CANYON-MED is a Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea [1]_ [2]_. When using this method, please cite the paper [1]_.

This model is available in **argopy** as an extension to the ``argo`` accessor: :class:`Dataset.argo.canyon_med`. It can be used to predict PO4, NO3, DIC, SiOH4, AT and pHT.

As an example, let's load one float data with oxygen measurements:

.. ipython:: python
    :okwarning:

    ArgoSet = DataFetcher(ds='bgc', mode='standard', params='DOXY', measured='DOXY').float(1902605)
    ds = ArgoSet.to_xarray()

We can then predict all possible variables:

.. ipython:: python
    :okwarning:

    ds.argo.canyon_med.predict()

or select variables to predict, like PO4:

.. ipython:: python
    :okwarning:

    ds = ds.argo.canyon_med.predict('PO4')
    ds['PO4']


.. _complement-optical-modeling:

Optical modeling
----------------

.. currentmodule:: xarray

This extension provides methods to compute standard variables from optical modeling of the upper ocean. This feature is available in **argopy** as an extension to the ``argo`` accessor: :class:`Dataset.argo.optic`. It can be used to:

- compute the depth of the euphotic zone, from PAR: :class:`Dataset.argo.optic.Zeu`
- compute the first optical depth, from depth of the euphotic zone: :class:`Dataset.argo.optic.Zpd`
- compute the depth where PAR reaches some threshold value: :class:`Dataset.argo.optic.Z_iPAR_threshold`
- search and qualify Deep Chlorophyll Maxima: :class:`Dataset.argo.optic.DCM`

As an example, let's load one BGC float data with DOWNWELLING_PAR measurements:

.. ipython:: python
    :okwarning:

    ds = DataFetcher(ds='bgc', mode='expert', params='DOWNWELLING_PAR').float(6901864).data
    dsp = ds.argo.point2profile()

.. currentmodule:: argopy

Note that we could have loaded these data with a :class:`ArgoFloat`:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(6901864).open_dataset('Sprof')

We can then simply call on the extension methods to add variables to the dataset:

.. ipython:: python
    :okwarning:

    dsp.argo.optic.Zeu()
    dsp.argo.optic.Zeu(method='percentage', max_surface=5.)
    dsp.argo.optic.Zeu(method='KdPAR', layer_min=10., layer_max=50.)

    dsp.argo.optic.Zpd()

    dsp.argo.optic.Z_iPAR_threshold(threshold=15.)

For the Deep Chlorophyll Maxima diagnostic, we need CHLA data, so let's load data from another BGC float:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(1902303).open_dataset('Sprof')
    dsp.argo.optic.DCM()

.. _perprofile-diag:

Per profile custom diagnostic
-----------------------------

.. currentmodule:: xarray

If you want to execute your own diagnostic method on a collection of Argo profiles, **argopy** provides an efficient method to do so, based on the :meth:`xarray.apply_ufunc` method.

The :meth:`Dataset.argo.reduce_profile` method allows to execute a per profile diagnostic function very efficiently. Such a diagnostic function takes vertical profiles as input and return a single value as output (see examples below). Typical usage example would include computation of mixed layer depth or euphotic layer depth.

In other words, the :meth:`Dataset.argo.reduce_profile` applies a vectorized function for unlabeled arrays on each Argo profiles loaded with **argopy**.

Diagnostic without option
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start with a trivial example. We load one BGC float synthetic data:

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(6901864).open_dataset('Sprof')

we then define a function that compute pressure of the salinity maximum. Note that this function deals with numpy or dask 1D arrays:

.. ipython:: python
    :okwarning:

    def max_salinity_depth(pres, psal):
        # A dummy function returning depth of the maximum salinity
        idx = ~np.logical_or(np.isnan(pres), np.isnan(psal))
        return pres[idx][np.argmax(psal[idx])]

and apply this function to all float profiles:

.. ipython:: python
    :okwarning:

    dsp.argo.reduce_profile(max_salinity_depth, params=['PRES', 'PSAL'])

**argopy** and xarray will handle all the axis and dimensions manipulation, so that you can focus on writing a *reducer* function dealing with 1D arrays for each requested parameters.

Diagnostic with option
^^^^^^^^^^^^^^^^^^^^^^

A more complex example will imply a function taking arguments. In the following we use the same trivial ``max_salinity_depth`` function as above, but provide a maximum depth argument.

.. ipython:: python
    :okwarning:

    def max_salinity_depth(pres, psal, max_layer=1000.):
        # A dummy function returning depth of the maximum salinity above max_layer:
        idx = ~np.logical_or(np.isnan(pres), np.isnan(psal))
        idx = np.logical_and(idx, pres<=max_layer)
        if np.any(idx):
            return pres[idx][np.argmax(psal[idx])]
        else:
            return np.NaN

Simply provide the function argument to the reducer:

.. ipython:: python
    :okwarning:

    dsp.argo.reduce_profile(max_salinity_depth,
                            params=['PRES', 'PSAL'],
                            max_layer=400.)

Optimisation with Dask
^^^^^^^^^^^^^^^^^^^^^^

At last, note that you can optmize this computation using Dask by simply chunking the dataset along the ``N_PROF`` dimension.

.. ipython:: python
    :okwarning:

    from argopy import ArgoFloat
    dsp = ArgoFloat(6901864).open_dataset('Sprof')

Make sure we're working with dask arrays:

.. ipython:: python
    :okwarning:

    dsp = dsp.chunk({'N_PROF': 10})

In this case, the :meth:`Dataset.argo.reduce_profile` will return a dask array:

.. ipython:: python
    :okwarning:

    da = dsp.argo.reduce_profile(max_salinity_depth, params=['PRES', 'PSAL'])
    da

And we need to trigger the computation to have it done:

.. ipython:: python
    :okwarning:

    da.compute()


References
----------
.. [1] Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2020). A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 7. doi:10.3389/fmars.2020.00620.

.. [2] Fourrier, M., Coppola, L., Claustre, H., D’Ortenzio, F., Sauzède, R., and Gattuso, J.-P. (2021). Corrigendum: A Regional Neural Network Approach to Estimate Water-Column Nutrient Concentrations and Carbonate System Variables in the Mediterranean Sea: CANYON-MED. Frontiers in Marine Science 8. doi:10.3389/fmars.2021.650509.
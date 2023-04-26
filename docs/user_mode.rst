.. _user-mode:

User mode: standard vs expert
=============================

**Problem**

For beginners or non-experts of the Argo dataset, it can be quite
complicated to get access to Argo measurements. Indeed, the Argo data
set is very complex, with thousands of different variables, tens of
reference tables and a `user manual <https://doi.org/10.13155/29825>`__
more than 100 pages long.

This is mainly due to:

-  Argo measurements coming from many different models of floats or
   sensors,
-  quality control of *in situ* measurements of autonomous platforms
   being really a matter of ocean and data experts,
-  the Argo data management workflow being distributed between more than
   10 Data Assembly Centers all around the world,
-  the Argo autonomous profiling floats, despite quite a simple
   principle of functioning, is a rather complex robot that needs a lot
   of data to be monitored and logged.

**Solution**

In order to ease Argo data analysis for the vast majority of standard
users, we implemented in **argopy** different levels of verbosity and
data processing to hide or simply remove variables only meaningful to
experts.

What type of user are you ?
---------------------------

If you don’t know in which user category you would place yourself, try
to answer the following questions:

-  what is a WMO number ?
-  what is the difference between Delayed and Real Time data mode ?
-  what is an adjusted parameter ?
-  what a QC flag of 3 means ?

If you answered to no more than 1 question, you probably would feel more
comfortable with the **standard** user mode. Otherwise, you can give a
try to the **expert** mode.

In **standard** mode, fetched data are automatically filtered to account
for their quality (only good are retained) and level of processing by
the data centers (whether they looked at the data briefly or not).

Setting the user mode
---------------------

Let's start with standard import:

.. ipython:: python
    :okwarning:

    import argopy
    from argopy import DataFetcher as ArgoDataFetcher


By default, all **argopy** data fetchers are set to work with a
**standard** user mode.

If you want to change the user mode, or simply makes it explicit, you
can use:

-  **argopy** global options:

.. ipython:: python
    :okwarning:

    argopy.set_options(mode='standard')

-  a temporary context:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='standard'):
        ArgoDataFetcher().profile(6902746, 34)

-  option when instantiating the data fetcher:

.. ipython:: python
    :okwarning:

    ArgoDataFetcher(mode='standard').profile(6902746, 34)

Differences in user modes
-------------------------

To highlight that, let’s compare data fetched for one profile with each
modes.

You will note that the **standard** mode has fewer variables to let you
focus on your analysis. For **expert**, all Argo variables for you to
work with are here.

The difference is the most visible when fetching Argo data from a local
copy of the GDAC ftp, so let’s use a sample of this provided by
**argopy** tutorial datasets:

.. ipython:: python
    :okwarning:

    ftproot, flist = argopy.tutorial.open_dataset('gdac')
    argopy.set_options(ftp=ftproot)

In **standard** mode:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='standard'):
        ds = ArgoDataFetcher(src='gdac').profile(6901929, 2).to_xarray()
        print(ds.data_vars)

In **expert** mode:

.. ipython:: python
    :okwarning:

    with argopy.set_options(mode='expert'):
        ds = ArgoDataFetcher(src='gdac').profile(6901929, 2).to_xarray()
        print(ds.data_vars)

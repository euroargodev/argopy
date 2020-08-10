**********************
Contributing to argopy
**********************

.. contents:: Table of contents:
   :local:

.. note::

  Large parts of this document came from the `Pandas Contributing
  Guide <http://pandas.pydata.org/pandas*docs/stable/contributing.html>`_.

If you seek **support** for your argopy usage: `visit the chat room at gitter <https://gitter.im/Argo-floats/argopy>`_.

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

We will complete this document for guidelines with regard to each of these contributions over time.

If you are brand new to *argopy* or open source development, we recommend going
through the `GitHub "issues" tab <https://github.com/euroargodev/argopy/issues>`_
to find issues that interest you. There are a number of issues listed under
`Documentation <https://github.com/euroargodev/argopy/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation>`_
and `good first issue
<https://github.com/euroargodev/argopy/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
where you could start out. Once you've found an interesting issue, you can
return here to get your development environment setup.


.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are an important part of making *argopy* more stable. Having a complete bug
report will allow others to reproduce the bug and provide insight into fixing. See
`this stackoverflow article <https://stackoverflow.com/help/mcve>`_ for tips on
writing a good bug report.

Trying the bug producing code out on the *master* branch is often a worthwhile exercise
to confirm the bug still exists. It is also worth searching existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self contained Python snippet reproducing the problem.
   You can format the code nicely by using `GitHub Flavored Markdown
   <http://github.github.com/github*flavored*markdown/>`_::

      ```python
      >>> import argopy as ar
      >>> ds = ar.DataFetcher(backend='erddap').float(5903248).to_xarray()
      ...
      ```

#. Include the full version string of *argopy* and its dependencies. You can use the
   built in function::

      >>> import argopy
      >>> argopy.show_versions()

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the :mod:`argopy` community and be open to comments/ideas
from others.

`Click here to open an issue with the specific bug reporting template <https://github.com/euroargodev/argopy/issues/new?template=bug_report.md>`_

.. _contributing.code:

Contributing to the code base
=============================

.. contents:: Code Base:
   :local:


Data fetchers
*************

Introduction
------------
If you want to add your own data fetcher for a new service, then, keep in mind that:

* Data fetchers are responsible for:

    * loading all available data from a given source and providing at least a :func:`to_xarray()` method
    * making data compliant to Argo standards (data type, variable name, attributes, etc ...)

* Data fetchers must:

    * inherit from the :class:`argopy.data_fetchers.proto.ArgoDataFetcherProto`
    * provide parameters:

            *  ``access_points``, eg: ['wmo', 'box']
            *  ``exit_formats``, eg: ['xarray']
            *  ``dataset_ids``, eg: ['phy', 'ref', 'bgc']

    * provides the facade API (:class:`argopy.fetchers.ArgoDataFetcher`) methods to filter data
    according to user level or requests. These must includes:


            *  :func:`filter_data_mode`
            *  :func:`filter_qc`
            *  :func:`filter_variables`


It is the responsability of the facade API (:class:`argopy.fetchers.ArgoDataFetcher`) to run
filters according to user level or requests, not the data fetcher.

Detailled guideline
-------------------

A new data fetcher must comply with:

Inheritance
~~~~~~~~~~~

-  [ ] Inherit from the :class:`argopy.data_fetchers.proto.ArgoDataFetcherProto`. This enforces minimal internal design
   compliance.

Auto-discovery for the facade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fetcher properties
^^^^^^^^^^^^^^^^^^

-  [ ] The new fetcher must come with the ``access_points``,
   ``exit_formats`` and ``dataset_ids`` properties at the top of the
   file, e.g.:

   ::

       access_points = ['wmo' ,'box']
       exit_formats = ['xarray']
       dataset_ids = ['phy', 'bgc']  # First is default

   Values depend on what the new access point can return and what you want to
   implement. A good start is with the ``wmo`` access point and the
   ``phy`` dataset ID. The ``xarray`` data format is the minimum
   required. These variables are used by the facade
   to auto-discover the fetcher capabilities. The ``dataset_ids``
   property is used to determine which variables can be retrieved.

Fetcher access points
^^^^^^^^^^^^^^^^^^^^^

-  [ ] The new fetcher must come at least with a ``Fetch_wmo`` or
   ``Fetch_wmo`` class, basically one for each of the ``access_points``
   listed as properties. More generaly we may have a main class that
   provides the key functionalities to retrieve data from the source,
   and then classes for each of the ``access_points`` of your fetcher.
   This pattern could look like this:

   ::

       class NewDataFetcher(ArgoDataFetcherProto)
       class Fetch_wmo(NewDataFetcher)
       class Fetch_box(NewDataFetcher)

   It could also be like:

   ::

       class Fetch_wmo(ArgoDataFetcherProto)
       class Fetch_box(ArgoDataFetcherProto)

   Note that the class names ``Fetch_wmo`` and ``Fetch_box`` must not
   change, this is also used by the facade to auto-discover the fetcher
   capabilities.

**Fetch\_wmo** is used to retrieve platforms and eventually profiles
data. It must take in the ``__init__()`` method a ``WMO`` and a ``CYC``
as first and second options. ``WMO`` is always passed, ``CYC`` is
optional. These are passed by the facade to implement the
``fetcher.float`` and ``fetcher.profile`` methods. When a float is requested, the ``CYC`` option is
not passed by the facade. Last, ``WMO`` and ``CYC`` are ether a single
integer or a list of integers: this means that ``Fetch_wmo`` must be
able to handle more than one float/platform retrieval.

**Fetch\_box** is used to retrieve a rectangular domain in space and
time. It must take in the ``__init__()`` method a ``BOX`` as first
option that is passed a list(lon\_min: float, lon\_max: float, lat\_min:
float, lat\_max: float, pres\_min: float, pres\_max: float, date\_min:
str, date\_max: str) from the facade. The two bounding dates [date\_min
and date\_max] should be optional (if not specified, the entire time
series is requested by the user).

File systems
~~~~~~~~~~~~

-  [ ] All http requests must go through the internal
   ``httpstore``, an internal wrapper around fsspec that allows to
   manage request caching very easily. You can simply use it this way
   for json requests:

   .. code:: python

       from argopy.stores import httpstore
       with httpstore(timeout=120).open("https://argovis.colorado.edu/catalog/profiles/5904797_12") as of:
           profile = json.load(of)

Output data format
~~~~~~~~~~~~~~~~~~

-  [ ] Last but not least, about the output data. In **argopy**, we want
   to provide data for both expert and standard users. This is explained
   and illustrated in the `documentation
   here <https://argopy.readthedocs.io/en/latest/user_mode.html>`__.
   This means for a new data fetcher that the data content
   should be curated and clean of any internal/jargon variables that is
   not part of the Argo ADMT vocabulary. For instance,
   variables like: ``bgcMeasKeys`` or ``geoLocation`` are not allowed. This will ensure
   that whatever the data source set by users, the output xarray or
   dataframe will be formatted and contain the same variables. This will
   also ensure that other argopy features can be used on the new fetcher
   output, like plotting or xarray data manipulation.


Code standards
**************

Writing good code is not just about what you write. It is also about *how* you
write it. During :ref:`Continuous Integration <contributing.ci>` testing, several
tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to *argopy*.


Code Formatting
~~~~~~~~~~~~~~~

*argopy* uses several tools to ensure a consistent code format throughout the project:

* `Flake8 <http://flake8.pycqa.org/en/latest/>`_ for general code quality

``pip``::

   pip install flake8

and then run from the root of the argopy repository::

   flake8

to qualify your code.
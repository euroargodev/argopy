**********************
Contributing to argopy
**********************

.. contents:: Table of contents:
   :local:

First off, thanks for taking the time to contribute!

.. note::

  Large parts of this document came from the `Xarray <http://xarray.pydata.org/en/stable/contributing.html>`_
  and `Pandas <http://pandas.pydata.org/pandas*docs/stable/contributing.html>`_ contributing guides.

If you seek **support** for your argopy usage or if you don't want to read
this whole thing and just have a question: `visit the chat room at gitter <https://gitter.im/Argo-floats/argopy>`_.

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

Please don't file an issue to ask a question, instead `visit the chat room at gitter <https://gitter.im/Argo-floats/argopy>`_.

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


.. _contributing.documentation:

Contributing to the documentation
=================================

If you're not the developer type, contributing to the documentation is still of
huge value. You don't even have to be an expert on *argopy* to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person.

.. contents:: Documentation:
   :local:


About the *argopy* documentation
--------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <http://sphinx-doc.org/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

- The *argopy* documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``argopy/docs/``.

  The docstrings are meant to provide a clear explanation of the usage of the
  individual functions, while the documentation in this folder consists of
  tutorial-like overviews per topic together with some other information
  (what's new, installation, etc).

- The docstrings follow the **Numpy Docstring Standard**, which is used widely
  in the Scientific Python community. This standard specifies the format of
  the different sections of the docstring. See `this document
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
  for a detailed explanation, or look at some of the existing functions to
  extend it in a similar manner.

- The tutorials make heavy use of the `ipython directive
  <http://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example:

  .. code:: rst

      .. ipython:: python

          x = 2
          x ** 3

  will be rendered as::

      In [1]: x = 2

      In [2]: x ** 3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make the doc building a bit more complex.

- Our API documentation in ``docs/api.rst`` houses the auto-generated
  documentation from the docstrings. For classes, there are a few subtleties
  around controlling which methods and attributes have pages auto-generated.

  Every method should be included in a ``toctree`` in ``api.rst``, else Sphinx
  will emit a warning.


How to build the *argopy* documentation
---------------------------------------

Requirements
^^^^^^^^^^^^
Make sure to follow the instructions on :ref:`creating a development environment below <contributing.dev_env>`, but
to build the docs you need to use the specific file ``docs/requirements.txt``:

.. code-block:: bash

    $ conda create --yes -n argopy-docs python=3.6 xarray dask numpy pytest future gsw sphinx sphinx_rtd_theme
    $ conda activate argopy-docs
    $ pip install argopy
    $ pip install -r docs/requirements.txt

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your local ``argopy/docs/`` directory in the console and run:

.. code-block:: bash

    make html

Then you can find the HTML output in the folder ``argopy/docs/_build/html/``.

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, sphinx will try to only build the pages that have been modified.

If you want to do a full clean build, do:

.. code-block:: bash

    make clean
    make html


.. _working.code:

Working with the code
=====================

Development workflow
--------------------

Anyone interested in helping to develop argopy needs to create their own fork
of our `git repository`. (Follow the github `forking instructions`_. You
will need a github account.)

.. _git repository: https://github.com/euroargodev/argopy
.. _forking instructions: https://help.github.com/articles/fork-a-repo/

Clone your fork on your local machine.

.. code-block:: bash

    $ git clone git@github.com:USERNAME/argopy

(In the above, replace USERNAME with your github user name.)

Then set your fork to track the upstream argopy repo.

.. code-block:: bash

    $ cd argopy
    $ git remote add upstream git://github.com/euroargodev/argopy.git

You will want to periodically sync your master branch with the upstream master.

.. code-block:: bash

    $ git fetch upstream
    $ git rebase upstream/master

**Never make any commits on your local master branch**. Instead open a feature
branch for every new development task.

.. code-block:: bash

    $ git checkout -b cool_new_feature

(Replace `cool_new_feature` with an appropriate description of your feature.)
At this point you work on your new feature, using `git add` to add your
changes. When your feature is complete and well tested, commit your changes

.. code-block:: bash

    $ git commit -m 'did a bunch of great work'

and push your branch to github.

.. code-block:: bash

    $ git push origin cool_new_feature

At this point, you go find your fork on github.com and create a `pull
request`_. Clearly describe what you have done in the comments. If your
pull request fixes an issue or adds a useful new feature, the team will
gladly merge it.

.. _pull request: https://help.github.com/articles/using-pull-requests/

After your pull request is merged, you can switch back to the master branch,
rebase, and delete your feature branch. You will find your new feature
incorporated into argopy.

.. code-block:: bash

    $ git checkout master
    $ git fetch upstream
    $ git rebase upstream/master
    $ git branch -d cool_new_feature

.. _contributing.dev_env:

Virtual environment
-------------------

This is how to create a virtual environment into which to test-install argopy,
install it, check the version, and tear down the virtual environment.

.. code-block:: bash

    $ conda create --yes -n argopy-tests python=3.6 xarray dask numpy pytest future gsw
    $ conda activate argopy-tests
    $ pip install argopy
    $ python -c 'import argopy; print(argopy.__version__);'
    $ conda deactivate
    $ conda env remove --yes -n argopy-tests


Code standards
--------------

Writing good code is not just about what you write. It is also about *how* you
write it. During Continuous Integration testing, several
tools will be run to check your code for stylistic errors.
Generating any warnings will cause the test to fail.
Thus, good style is a requirement for submitting code to *argopy*.

Code Formatting
---------------

*argopy* uses several tools to ensure a consistent code format throughout the project:

* `Flake8 <http://flake8.pycqa.org/en/latest/>`_ for general code quality

``pip``::

   pip install flake8

and then run from the root of the argopy repository::

   flake8

to qualify your code.


.. _contributing.code:

Contributing to the code base
=============================

.. contents:: Code Base:
   :local:

Data fetchers
-------------

Introduction
^^^^^^^^^^^^
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


It is the responsibility of the facade API (:class:`argopy.fetchers.ArgoDataFetcher`) to run
filters according to user level or requests, not the data fetcher.

Detailed guideline
^^^^^^^^^^^^^^^^^^

A new data fetcher must comply with:

Inheritance
"""""""""""

Inherit from the :class:`argopy.data_fetchers.proto.ArgoDataFetcherProto`.
This enforces minimal internal design compliance.

Auto-discovery of fetcher properties
""""""""""""""""""""""""""""""""""""

The new fetcher must come with the ``access_points``, ``exit_formats`` and ``dataset_ids`` properties at the top of the
file, e.g.:

.. code-block:: python

    access_points = ['wmo' ,'box']
    exit_formats = ['xarray']
    dataset_ids = ['phy', 'bgc']  # First is default

Values depend on what the new access point can return and what you want to
implement. A good start is with the ``wmo`` access point and the
``phy`` dataset ID. The ``xarray`` data format is the minimum
required. These variables are used by the facade
to auto-discover the fetcher capabilities. The ``dataset_ids``
property is used to determine which variables can be retrieved.

Auto-discovery of fetcher access points
"""""""""""""""""""""""""""""""""""""""

The new fetcher must come at least with a ``Fetch_box`` or
``Fetch_wmo`` class, basically one for each of the ``access_points``
listed as properties. More generally we may have a main class that
provides the key functionality to retrieve data from the source,
and then classes for each of the ``access_points`` of your fetcher.
This pattern could look like this:

.. code-block:: python

    class NewDataFetcher(ArgoDataFetcherProto)
    class Fetch_wmo(NewDataFetcher)
    class Fetch_box(NewDataFetcher)

It could also be like:

.. code-block:: python

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
""""""""""""

All http requests must go through the internal
``httpstore``, an internal wrapper around fsspec that allows to
manage request caching very easily. You can simply use it this way
for json requests:

.. code-block:: python

    from argopy.stores import httpstore
    with httpstore(timeout=120).open("https://argovis.colorado.edu/catalog/profiles/5904797_12") as of:
       profile = json.load(of)

Output data format
""""""""""""""""""

Last but not least, about the output data. In **argopy**, we want
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


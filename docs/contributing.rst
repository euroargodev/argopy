**********************
Contributing to argopy
**********************

.. contents:: Table of contents:
   :local:

.. note::

  Large parts of this document came from the `Pandas Contributing
  Guide <http://pandas.pydata.org/pandas*docs/stable/contributing.html>`_.

Where to start?
===============

All contributions, bug reports, bug fixes, documentation improvements,
enhancements, and ideas are welcome.

We will complete this document for guidelines with regard to each of these contributions over time.

If you are brand new to *argopy* or open*source development, we recommend going
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

If you want to add your own data fetcher from a new service, then, keep in mind that:

* Data fetchers are responsible for:

    * loading all available data from a given source and providing at least a :func:`to_xarray()` method
    * making data compliant to Argo standards (data type, variable name, attributes, etc ...)


* Data fetchers must:

    * inherit from the `argopy.data_fetchers.proto.ArgoDataFetcherProto`
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
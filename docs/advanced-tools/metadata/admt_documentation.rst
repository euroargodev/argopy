.. currentmodule:: argopy.related

ADMT Documentation
------------------

More than 20 pdf manuals have been produced by the Argo Data Management Team. Using the :class:`ArgoDocs` class, it's easy to navigate this great database.

If you don't know where to start, you can simply list all available documents:

.. ipython:: python
    :okwarning:

    from argopy import ArgoDocs

    ArgoDocs().list

Or search for a word in the title and/or abstract:

.. ipython:: python
    :okwarning:

    results = ArgoDocs().search("oxygen")
    for docid in results:
        print("\n", ArgoDocs(docid))

Then using the Argo doi number of a document, you can easily retrieve it:

.. ipython:: python
    :okwarning:

    ArgoDocs(35385)

and open it in your browser:

.. ipython:: python
    :okwarning:

    # ArgoDocs(35385).show()
    # ArgoDocs(35385).open_pdf(page=12)

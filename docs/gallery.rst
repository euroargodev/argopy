Gallery
=======

Here's a list of examples on how to use **argopy**. We will be adding more examples soon.
Contributions are highly welcomed and appreciated. So, if you are interested in contributing, please consult the
:doc:`contributing` guide.


Notebook Examples
-----------------

.. grid:: 1 1 2 2
    :gutter: 4

    .. grid-item-card:: BGC one float data
        :img-top: _static/nb_examples_one_float_data.png

        |select_float| |ds_bgc| |mode_expert| |src_erddap|
        ^^^
        A notebook to download and plot one BGC float data
        +++
        .. grid:: 2 2 2 2

            .. grid-item::

                .. button-link:: https://nbviewer.org/github/euroargodev/argopy/blob/master/docs/examples/BGC_one_float_data.ipynb
                    :color: primary
                    :outline:

                    :fas:`eye` Online viewer

            .. grid-item::

                .. button-link:: https://www.github.com/euroargodev/argopy/blob/master/docs/examples/BGC_one_float_data.ipynb
                    :color: primary
                    :outline:

                    :fas:`file-arrow-down` Download notebook

    .. grid-item-card:: BGC regional data
        :img-top: _static/nbexamples_region_bgc_data.png

        |select_region| |ds_bgc| |mode_expert| |src_erddap|
        ^^^
        A notebook to download and plot BGC data in a specific ocean region
        +++
        .. grid:: 2 2 2 2

            .. grid-item::

                .. button-link:: https://nbviewer.org/github/euroargodev/argopy/blob/master/docs/examples/BGC_region_float_data.ipynb
                    :color: primary
                    :outline:

                    :fas:`eye` Online viewer

            .. grid-item::

                .. button-link:: https://www.github.com/euroargodev/argopy/blob/master/docs/examples/BGC_region_float_data.ipynb
                    :color: primary
                    :outline:

                    :fas:`file-arrow-down` Download notebook

    .. grid-item-card:: Scatter map with data mode of one BGC variable
        :img-top: _static/nbexamples_scatter_map_datamode_bgc.png

        |select_region| |select_float| |ds_bgc| |mode_expert| |src_gdac|
        ^^^
        A notebook to plot a global map where profile locations are color coded with one BGC parameter data mode
        +++
        .. grid:: 2 2 2 2

            .. grid-item::

                .. button-link:: https://nbviewer.org/github/euroargodev/argopy/blob/master/docs/examples/BGC_scatter_map_data_mode.ipynb
                    :color: primary
                    :outline:

                    :fas:`eye` Online viewer

            .. grid-item::

                .. button-link:: https://www.github.com/euroargodev/argopy/blob/master/docs/examples/BGC_scatter_map_data_mode.ipynb
                    :color: primary
                    :outline:

                    :fas:`file-arrow-down` Download notebook

    .. grid-item-card:: BGC data mode census
        :img-top: _static/nbexamples_datamode_bgc_census.png

        |select_region| |ds_bgc| |mode_expert| |src_gdac|
        ^^^
        A notebook to make a global census of all BGC parameter data mode and a pie plot with results
        +++
        .. grid:: 2 2 2 2

            .. grid-item::

                .. button-link:: https://nbviewer.org/github/euroargodev/argopy/blob/master/docs/examples/BGC_data_mode_census.ipynb
                    :color: primary
                    :outline:

                    :fas:`eye` Online viewer

            .. grid-item::

                .. button-link:: https://www.github.com/euroargodev/argopy/blob/master/docs/examples/BGC_data_mode_census.ipynb
                    :color: primary
                    :outline:

                    :fas:`file-arrow-down` Download notebook

.. dropdown:: Notebook tags Legend
    :open:

    :Data selection: |select_region| : region, |select_float| : float, |select_profile| : profile
    :Dataset: |ds_phy| : core+deep, |ds_bgc| : BGC
    :User mode: |mode_expert| : expert, |mode_standard| : standard, |mode_research| : research
    :Data sources: |src_erddap| : erddap, |src_gdac| : gdac, |src_argovis| : argovis


.. |src_erddap| replace:: ⭐
.. |src_gdac| replace:: 🌐
.. |src_argovis| replace:: 👁
.. |ds_phy| replace:: 🟡+🔵
.. |ds_bgc| replace:: 🟢
.. |mode_expert| replace:: 🏄
.. |mode_standard| replace:: 🏊
.. |mode_research| replace:: 🚣
.. |select_region| replace:: 🗺
.. |select_float| replace:: 🤖
.. |select_profile| replace:: ⚓

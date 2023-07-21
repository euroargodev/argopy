Gallery
=======

Here's a list of examples on how to use **argopy**. We will be adding more examples soon.
Contributions are highly welcomed and appreciated. So, if you are interested in contributing, please consult the
:doc:`contributing` guide.


Notebook Examples
-----------------

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: Scatter map with data mode of one BGC variable
        :link: https://github.com/euroargodev/argopy/blob/master/docs/examples/scatter_map_BGC_data_mode.ipynb
        :link-type: url
        :img-top: _static/nbexamples_scatter_map_datamode_bgc.png

        |select_region| |select_float| |ds_bgc| |mode_expert| |src_gdac|
        ^^^
        A notebook to plot a map where profile locations are color coded with one BGC parameter data mode

    .. grid-item-card:: BGC data mode census
        :link: https://github.com/euroargodev/argopy/blob/master/docs/examples/BGC_data_mode_census.ipynb
        :link-type: url
        :img-top: _static/nbexamples_datamode_bgc_census.png

        |select_region| |ds_bgc| |mode_expert| |src_gdac|
        ^^^
        A notebook to make a census of all BGC parameter data mode and a pie plot with results


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
.. |other| image:: https://img.shields.io/static/v1?label=source&message=erddap&color=blue

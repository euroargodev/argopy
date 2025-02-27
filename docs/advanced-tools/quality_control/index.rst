.. _data_qc:

Data quality control
====================

Quality control in Argo is a complex and a big deal. Most experts and operators have their own procedures to perform QC.
That's why in **argopy** we're trying to provide tools complementary to existing procedures and elementary blocks to
elaborate a python-based QC pipeline for Argo. These are:

* :doc:`Argo file stores <../stores/index>` to separate I/O operations from data analysis
* :doc:`Data preprocessing for the OWC salinity calibration <salinity>`
* :doc:`Facilitated access to reference dataset <reference_dataset>`
* :doc:`Topographic data for trajectory analysis <trajectories>`
* :doc:`Easy access to CLS altimetry QC test results <altimetry>`

.. toctree::
    :maxdepth: 2
    :hidden:

    salinity
    reference_dataset
    trajectories
    altimetry

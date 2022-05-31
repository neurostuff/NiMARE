API
===

.. _api_dataset_ref:

:mod:`nimare.dataset`: Dataset IO
--------------------------------------------------

.. automodule:: nimare.dataset
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dataset.Dataset
   dataset.DatasetSearcher


.. _api_meta_ref:

:mod:`nimare.meta`: Meta-analytic algorithms
--------------------------------------------------

For more information about the components of coordinate-based meta-analysis in NiMARE, see :doc:`cbma`.

.. automodule:: nimare.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   meta.ibma
   meta.cbma.ale
   meta.cbma.mkda
   meta.cbma.base
   meta.kernel

.. _api_results_ref:

:mod:`nimare.results`: Meta-analytic results
------------------------------------------------------

.. automodule:: nimare.results
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   results.MetaResult


.. _api_correct_ref:

:mod:`nimare.correct`: Multiple comparisons correction
------------------------------------------------------

.. automodule:: nimare.correct
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   correct.FWECorrector
   correct.FDRCorrector


.. _api_diagnostics_ref:

:mod:`nimare.diagnostics`: Diagnostics
------------------------------------------------------

.. automodule:: nimare.diagnostics
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   diagnostics.Jackknife
   diagnostics.FocusCounter


.. _api_annotate_ref:

:mod:`nimare.annotate`: Automated annotation
--------------------------------------------------

.. automodule:: nimare.annotate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   annotate.cogat
   annotate.gclda
   annotate.lda
   annotate.text
   annotate.utils


.. _api_decode_ref:

:mod:`nimare.decode`: Functional characterization analysis
-----------------------------------------------------------
For more information about functional characterization analysis, see :doc:`decoding`.

.. automodule:: nimare.decode
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   decode.discrete
   decode.continuous
   decode.encode
   decode.base


.. _api_io_ref:

:mod:`nimare.io`: Tools for ingesting data in other formats
-----------------------------------------------------------

.. automodule:: nimare.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   io.convert_neurosynth_to_dict
   io.convert_neurosynth_to_json
   io.convert_neurosynth_to_dataset
   io.convert_sleuth_to_dict
   io.convert_sleuth_to_json
   io.convert_sleuth_to_dataset
   io.convert_neurovault_to_dataset


.. _api_transforms_ref:

:mod:`nimare.transforms`: Data transforms
-----------------------------------------------------

.. automodule:: nimare.transforms
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transforms.ImageTransformer
   transforms.ImagesToCoordinates

   :template: function.rst

   transforms.transform_images
   transforms.resolve_transforms
   transforms.sample_sizes_to_dof
   transforms.sample_sizes_to_sample_size
   transforms.sd_to_varcope
   transforms.se_to_varcope
   transforms.samplevar_dataset_to_varcope
   transforms.t_and_varcope_to_beta
   transforms.t_and_beta_to_varcope
   transforms.p_to_z
   transforms.t_to_z
   transforms.z_to_t
   transforms.z_to_p


.. _api_extract_ref:

:mod:`nimare.extract`: Dataset and model fetching
-----------------------------------------------------
For more information about fetching data from the internet, see :ref:`fetching tools`.

.. automodule:: nimare.extract
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   extract.fetch_neuroquery
   extract.fetch_neurosynth
   extract.download_nidm_pain
   extract.download_cognitive_atlas
   extract.download_abstracts
   extract.download_peaks2maps_model

   extract.utils.get_data_dirs


.. _api_stats_ref:

:mod:`nimare.stats`: Statistical functions
-----------------------------------------------------

.. automodule:: nimare.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   stats.one_way
   stats.two_way
   stats.pearson
   stats.null_to_p
   stats.nullhist_to_p
   stats.fdr


.. _api_generate_ref:

:mod:`nimare.generate`: Data generation functions
-----------------------------------------------------

.. automodule:: nimare.generate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   generate.create_coordinate_dataset
   generate.create_neurovault_dataset


.. _api_utils_ref:

:mod:`nimare.utils`: Utility functions and submodules
-----------------------------------------------------

.. automodule:: nimare.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.get_template
   utils.get_masker
   utils.get_resource_path
   utils.vox2mm
   utils.mm2vox
   utils.tal2mni
   utils.mni2tal

.. _api_workflows_ref:

:mod:`nimare.workflows`: Common workflows
--------------------------------------------------

.. automodule:: nimare.workflows
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   workflows.ale_sleuth_workflow
   workflows.conperm_workflow
   workflows.macm_workflow
   workflows.peaks2maps_workflow
   workflows.scale_workflow


.. _api_base_ref:

:mod:`nimare.base`: Base classes
--------------------------------------------------
.. automodule:: nimare.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.NiMAREBase
   base.Estimator

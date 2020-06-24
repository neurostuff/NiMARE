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


.. _api_meta_ref:

:mod:`nimare.meta`: Meta-analytic algorithms
--------------------------------------------------

.. automodule:: nimare.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   meta.esma
   meta.ibma
   meta.cbma.kernel
   meta.cbma.ale
   meta.cbma.mkda

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


.. _api_io_ref:

:mod:`nimare.io`: Input/Output
-----------------------------------------------------

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


.. _api_transforms_ref:

:mod:`nimare.transforms`: Data transforms
-----------------------------------------------------

.. automodule:: nimare.transforms
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
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
   transforms.vox2mm
   transforms.mm2vox
   transforms.tal2mni
   transforms.mni2tal


.. _api_extract_ref:

:mod:`nimare.extract`: Dataset and model fetching
-----------------------------------------------------

.. automodule:: nimare.extract
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   extract.download_nidm_pain
   extract.download_mallet
   extract.download_cognitive_atlas
   extract.download_abstracts
   extract.download_peaks2maps_model


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
   stats.fdr


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

   utils.dict_to_df
   utils.dict_to_coordinates
   utils.validate_df
   utils.validate_images_df
   utils.get_template
   utils.get_masker
   utils.listify
   utils.round2
   utils.get_resource_path
   utils.find_stem
   utils.uk_to_us


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
   base.MetaEstimator
   base.CBMAEstimator
   base.Transformer
   base.KernelTransformer
   base.Decoder

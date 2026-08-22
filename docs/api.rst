API
===

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
   meta.cbmr

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

   diagnostics.FocusFilter
   diagnostics.Jackknife
   diagnostics.FocusCounter
   diagnostics.ResampledStability


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
   io.fetch_neurostore_studyset
   io.convert_nimads_to_dataset
   io.convert_nimads_to_sleuth
   io.convert_dataset_to_nimads_dict
   io.convert_dataset_to_studyset
   io.convert_sleuth_to_dict
   io.convert_sleuth_to_json
   io.convert_sleuth_to_dataset
   io.convert_sleuth_to_studyset
   io.convert_neurovault_to_dataset


.. _api_nimads_ref:

:mod:`nimare.studyset`: NeuroImaging Meta-Analysis Data Structure
------------------------------------------------------------------

The NIMADS :class:`~nimare.studyset.Studyset` is the primary collection type for
all NiMARE workflows. :mod:`nimare.nimads` re-exports this module under the
historical import path and adds nothing of its own.

.. automodule:: nimare.studyset
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   studyset.Studyset
   studyset.StudysetStore
   studyset.View
   studyset.Context
   studyset.Study
   studyset.Analysis
   studyset.Image
   studyset.Point
   studyset.AnnotationSet
   studyset.ColumnStore
   studyset.Coordinates
   studyset.Images
   studyset.Labels
   studyset.Texts
   studyset.PerAnalysis
   studyset.CoordinateBlock
   studyset.ImageBlock
   studyset.LabelBlock
   studyset.TextBlock
   studyset.Comparison
   studyset.normalize_collection
   studyset.check_invariants
   studyset.from_nimads
   studyset.from_parquet
   studyset.write_nimads
   studyset.write_parquet
   studyset.convert_neurostore_json_to_parquet


.. _api_dataset_ref:

:mod:`nimare.dataset`: Legacy Dataset IO
--------------------------------------------------

.. warning::
    :class:`~nimare.dataset.Dataset` is deprecated and will be removed in NiMARE 1.0.0.
    Constructing one, or passing one to any algorithm, raises a :class:`FutureWarning`.
    Use :class:`~nimare.studyset.Studyset` instead -- see
    :meth:`~nimare.studyset.Studyset.from_dataset` and the
    ``nimare.io.convert_*_to_studyset`` functions for the migration paths.

.. automodule:: nimare.dataset
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dataset.Dataset


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
   transforms.StandardizeField

   :template: function.rst

   transforms.threshold_image
   transforms.transform_images
   transforms.resolve_transforms
   transforms.sample_sizes_to_dof
   transforms.sample_sizes_to_sample_size
   transforms.sd_to_varcope
   transforms.se_to_varcope
   transforms.samplevar_dataset_to_varcope
   transforms.t_and_varcope_to_beta
   transforms.t_and_beta_to_varcope
   transforms.t_to_d
   transforms.d_to_g
   transforms.p_to_z
   transforms.nlogp_to_z
   transforms.t_to_z
   transforms.z_to_t
   transforms.z_to_p
   transforms.z_to_nlogp
   transforms.t_to_nlogp
   transforms.chi2_to_nlogp


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
   stats.two_way_counts
   stats.two_way
   stats.pearson
   stats.null_to_p
   stats.nullhist_to_p
   stats.nlogp_bonferroni
   stats.nlogp_fdr


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
   generate.create_coordinate_studyset
   generate.create_neurovault_dataset
   generate.create_neurovault_studyset


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
   utils.unique_rows

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

   workflows.macm_workflow
   workflows.cbma.CBMAWorkflow
   workflows.cbma.ContrastWorkflow
   workflows.cbma.PairwiseCBMAWorkflow
   workflows.ibma.IBMAWorkflow
   workflows.misc.conjunction_analysis

:mod:`nimare.reports`: NiMARE report
--------------------------------------------------

.. automodule:: nimare.reports
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: function.rst

   reports.run_reports

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
   estimator.Estimator

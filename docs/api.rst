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
   meta.cbma.model

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
   annotate.cogpo
   annotate.utils
   annotate.boltzmann
   annotate.gclda
   annotate.lda
   annotate.text2brain
   annotate.word2brain
   annotate.text


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


.. _api_parcellate_ref:

:mod:`nimare.parcellate`: Meta-analytic parcellation
----------------------------------------------------

.. automodule:: nimare.parcellate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   parcellate.cbp
   parcellate.mamp
   parcellate.mapbot


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
   stats.p_to_z
   stats.t_to_z
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

   utils.get_template
   utils.listify
   utils.round2
   utils.vox2mm
   utils.mm2vox
   utils.tal2mni
   utils.mni2tal
   utils.get_resource_path


.. _api_workflows_ref:

:mod:`nimare.workflows`: Common workflows
--------------------------------------------------

.. automodule:: nimare.workflows
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare

.. autosummary::
   :toctree: generated/
   :template: module.rst

   workflows.ale
   workflows.conperm
   workflows.macm
   workflows.peaks2maps
   workflows.scale


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
   base.Transformer

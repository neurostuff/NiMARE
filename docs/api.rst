API
===

.. _api_dataset_ref:

:mod:`nimare.dataset`: Dataset IO
--------------------------------------------------

.. automodule:: nimare.dataset
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.dataset

.. autosummary:: nimare.dataset
   :toctree: generated/
   :template: class.rst

   nimare.dataset.Dataset


.. _api_meta_ref:

:mod:`nimare.meta`: Meta-analytic algorithms
--------------------------------------------------

.. automodule:: nimare.meta
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.meta

.. autosummary:: nimare.meta
   :toctree: generated/
   :template: module.rst

   nimare.meta.esma
   nimare.meta.ibma
   nimare.meta.cbma.kernel
   nimare.meta.cbma.ale
   nimare.meta.cbma.mkda
   nimare.meta.cbma.model
   nimare.meta.base

.. _api_results_ref:

:mod:`nimare.results`: Meta-analytic results
------------------------------------------------------

.. automodule:: nimare.results
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.results

.. autosummary:: nimare.results
   :toctree: generated/
   :template: class.rst

   nimare.results.MetaResult


.. _api_correct_ref:

:mod:`nimare.correct`: Multiple comparisons correction
------------------------------------------------------

.. automodule:: nimare.correct
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.correct

.. autosummary:: nimare.correct
   :toctree: generated/
   :template: class.rst

   nimare.correct.FWECorrector
   nimare.correct.FDRCorrector


.. _api_annotate_ref:

:mod:`nimare.annotate`: Automated annotation
--------------------------------------------------

.. automodule:: nimare.annotate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.annotate

.. autosummary:: nimare.annotate
   :toctree: generated/
   :template: module.rst

   nimare.annotate.ontology
   nimare.annotate.topic
   nimare.annotate.vector
   nimare.annotate.text


.. _api_decode_ref:

:mod:`nimare.decode`: Functional characterization analysis
-----------------------------------------------------------

.. automodule:: nimare.decode
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.decode

.. autosummary:: nimare.decode
   :toctree: generated/
   :template: module.rst

   nimare.decode.discrete
   nimare.decode.continuous
   nimare.decode.encode


.. _api_parcellate_ref:

:mod:`nimare.parcellate`: Meta-analytic parcellation
----------------------------------------------------

.. automodule:: nimare.parcellate
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.parcellate

.. autosummary:: nimare.parcellate
   :toctree: generated/
   :template: module.rst

   nimare.parcellate.cbp
   nimare.parcellate.mamp
   nimare.parcellate.mapbot


.. _api_io_ref:

:mod:`nimare.io`: Input/Output
-----------------------------------------------------

.. automodule:: nimare.io
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.io

.. autosummary:: nimare.io
   :toctree: generated/
   :template: function.rst

   nimare.io.convert_neurosynth_to_dict
   nimare.io.convert_neurosynth_to_json
   nimare.io.convert_neurosynth_to_dataset
   nimare.io.convert_sleuth_to_dict
   nimare.io.convert_sleuth_to_json
   nimare.io.convert_sleuth_to_dataset


.. _api_extract_ref:

:mod:`nimare.extract`: Dataset and model fetching
-----------------------------------------------------

.. automodule:: nimare.extract
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.extract

.. autosummary:: nimare.extract
   :toctree: generated/
   :template: function.rst

   nimare.extract.download_nidm_pain
   nimare.extract.download_mallet
   nimare.extract.download_cognitive_atlas
   nimare.extract.download_abstracts


.. _api_stats_ref:

:mod:`nimare.stats`: Statistical functions
-----------------------------------------------------

.. automodule:: nimare.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.stats

.. autosummary:: nimare.stats
   :toctree: generated/
   :template: function.rst

   nimare.stats.one_way
   nimare.stats.two_way
   nimare.stats.pearson
   nimare.stats.null_to_p
   nimare.stats.p_to_z
   nimare.stats.t_to_z
   nimare.stats.fdr


.. _api_utils_ref:

:mod:`nimare.utils`: Utility functions and submodules
-----------------------------------------------------

.. automodule:: nimare.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.utils

.. autosummary:: nimare.utils
   :toctree: generated/
   :template: function.rst

   nimare.utils.get_template
   nimare.utils.listify
   nimare.utils.round2
   nimare.utils.vox2mm
   nimare.utils.mm2vox
   nimare.utils.tal2mni
   nimare.utils.mni2tal
   nimare.utils.get_resource_path


.. _api_workflows_ref:

:mod:`nimare.workflows`: Common workflows
--------------------------------------------------

.. automodule:: nimare.workflows
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.workflows

.. autosummary:: nimare.workflows
   :toctree: generated/
   :template: module.rst

   nimare.workflows.ale
   nimare.workflows.conperm
   nimare.workflows.macm
   nimare.workflows.peaks2maps
   nimare.workflows.scale


.. _api_base_ref:

:mod:`nimare.base`: Base classes
--------------------------------------------------
.. automodule:: nimare.base
   :no-members:
   :no-inherited-members:

.. currentmodule:: nimare.base

.. autosummary:: nimare.base
   :toctree: generated/
   :template: class.rst

   nimare.base.NiMAREBase
   nimare.base.Estimator
   nimare.base.Transformer

.. _examples-metas:

Performing meta-analyses
------------------------

NiMARE implements a number of coordinate- and image-based meta-analysis algorithms in its :mod:`~nimare.meta` module.
The examples below use :class:`~nimare.nimads.Studyset` as the primary analysis input,
with legacy :class:`~nimare.dataset.Dataset` objects appearing only in preprocessing steps
where older APIs still require them.

For more information about the components that go into coordinate-based meta-analyses in NiMARE, see :doc:`../cbma`,
as well as :doc:`../outputs`.

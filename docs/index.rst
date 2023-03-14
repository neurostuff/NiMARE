.. include:: <isonum.txt>

NiMARE: Neuroimaging Meta-Analysis Research Environment
==========================================================

NiMARE is a Python package for neuroimaging meta-analyses.
It makes conducting scary meta-analyses a dream!

To install NiMARE check out our `installation guide`_.

.. image:: https://img.shields.io/pypi/v/nimare.svg
   :target: https://pypi.python.org/pypi/nimare/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/nimare.svg
   :target: https://pypi.python.org/pypi/nimare/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/Source%20Code-neurostuff%2Fnimare-purple
   :target: https://github.com/neurostuff/NiMARE
   :alt: GitHub Repository

.. image:: https://zenodo.org/badge/117724523.svg
   :target: https://zenodo.org/badge/latestdoi/117724523
   :alt: DOI

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://github.com/neurostuff/NiMARE/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/neurostuff/NiMARE/actions/workflows/testing.yml
   :alt: Test Status

.. image:: https://readthedocs.org/projects/nimare/badge/?version=latest
   :target: http://nimare.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://codecov.io/gh/neurostuff/NiMARE/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/neurostuff/nimare
   :alt: Codecov

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://img.shields.io/badge/mattermost-join_chat%20%E2%86%92-brightgreen.svg
   :target: https://mattermost.brainhack.org/brainhack/channels/nimare
   :alt: Join the chat

.. image:: https://img.shields.io/badge/RRID-SCR__017398-blue.svg
   :target: https://scicrunch.org/scicrunch/Resources/record/nlx_144509-1/SCR_017398/resolver?q=nimare&l=nimare
   :alt: RRID:SCR_017398

.. image:: http://neurolibre.herokuapp.com/papers/10.55458/neurolibre.00007/status.svg
   :target: https://doi.org/10.55458/neurolibre.00007
   :alt: NeuroLibre preprint

.. _installation guide: installation.html

.. image:: _static/nimare_overview.png
   :align: center

Citing NiMARE
-------------

If you use NiMARE in your research, we recommend citing the Zenodo DOI associated with the NiMARE version you used,
as well as the NeuroLibre preprint for the NiMARE Jupyter book.
You can find the Zenodo DOI associated with each NiMARE release at https://zenodo.org/record/6642243#.YqiXNy-B1KM.

.. code-block:: bibtex
   :caption: BibTeX entries for NiMARE version 0.0.11.

   # This is the NeuroLibre preprint.
   @article{Salo2022,
   doi = {10.55458/neurolibre.00007},
   url = {https://doi.org/10.55458/neurolibre.00007},
   year = {2022},
   publisher = {The Open Journal},
   volume = {1},
   number = {1},
   pages = {7},
   author = {Taylor Salo and Tal Yarkoni and Thomas E. Nichols and Jean-Baptiste Poline and Murat Bilgel and Katherine L. Bottenhorn and Dorota Jarecka and James D. Kent and Adam Kimbler and Dylan M. Nielson and Kendra M. Oudyk and Julio A. Peraza and Alexandre Pérez and Puck C. Reeders and Julio A. Yanes and Angela R. Laird},
   title = {NiMARE: Neuroimaging Meta-Analysis Research Environment},
   journal = {NeuroLibre}
   }

   # This is the Zenodo citation for version 0.0.11.
   @software{salo_taylor_2022_5826281,
   author       = {Salo, Taylor and
                     Yarkoni, Tal and
                     Nichols, Thomas E. and
                     Poline, Jean-Baptiste and
                     Kent, James D. and
                     Gorgolewski, Krzysztof J. and
                     Glerean, Enrico and
                     Bottenhorn, Katherine L. and
                     Bilgel, Murat and
                     Wright, Jessey and
                     Reeders, Puck and
                     Kimbler, Adam and
                     Nielson, Dylan N. and
                     Yanes, Julio A. and
                     Pérez, Alexandre and
                     Oudyk, Kendra M. and
                     Jarecka, Dorota and
                     Enge, Alexander and
                     Peraza, Julio A. and
                     Laird, Angela R.},
   title        = {neurostuff/NiMARE: 0.0.11},
   month        = jan,
   year         = 2022,
   publisher    = {Zenodo},
   version      = {0.0.11},
   doi          = {10.5281/zenodo.5826281},
   url          = {https://doi.org/10.5281/zenodo.5826281}
   }

Then, to cite NiMARE in your manuscript, we recommend something like the following:

   We used NiMARE v0.0.11 (RRID:SCR_017398; Salo et al., 2022a; Salo et al., 2022b).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   installation
   api
   auto_examples/index
   contributing
   dev_guide
   cli
   outputs
   methods
   changelog
   glossary

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

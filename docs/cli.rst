Command Line Interface
========================

NiMARE provides several workflows as command-line interfaces, including ALE
meta-analysis, meta-analytic coactivation modeling (MACM) analysis, peaks2maps
image reconstruction, and contrast map meta-analysis.
Each workflow should generate a boilerplate paragraph with details about the
workflow and citations that can be used in a manuscript.

To use NiMARE from the command line, open a terminal window and type:

.. code-block:: bash

	nimare --help

This will print the instructions for using the command line interface in your
command line.

.. argparse::
   :ref: nimare.cli._get_parser
   :prog: nimare
   :func: _get_parser

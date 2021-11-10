.. include:: links.rst

NiMARE Developer Guide
======================

This guide provides a more detailed description of the organization and preferred coding style for NiMARE, for prospective code contributors.

Coding Style
------------

NiMARE code should follow PEP8 recommendations.

To enforce NiMARE's preferred coding style,
we use `flake8`_ with plugins for `isort <https://pypi.org/project/flake8-isort/>`_,
`black <https://pypi.org/project/flake8-black/>`_, and `docstrings <https://pypi.org/project/flake8-docstrings/>`_.
These plugins automatically evaluate imports, code formatting, and docstring formatting as part of our continuous integraton.

Additionally, we have modeled NiMARE's code on `scikit-learn`_.
By this we mean that most of NiMARE user-facing tools are implemented as classes.
These classes generally accept a number of parameters at initialization,
and then use ``fit`` or ``transform`` methods to apply the algorithm to data (generally a NiMARE ``Dataset`` object).

Installation with Docker
------------------------

You may wish to use Docker to control your environment when testing or developing on NiMARE.
Here are some common steps for taking this approach:

To build the Docker image:

.. code-block:: bash

  docker build -t test/nimare .

To run the Docker container:

.. code-block:: bash

  docker run -it -v `pwd`:/home/neuro/code/NiMARE -p8888:8888 test/nimare bash

Once inside the container, you can install NiMARE:

.. code-block:: bash

  python /home/neuro/code/NiMARE/setup.py develop

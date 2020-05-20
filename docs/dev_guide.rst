.. include:: links.rst

NiMARE Developer Guide
======================

This guide provides a more detailed description of the organization and preferred coding style for NiMARE, for prospective code contributors.

Coding Style
------------
NiMARE code should follow PEP8 recommendations.
Additionally, we have modeled NiMARE's code on `scikit-learn`_.

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

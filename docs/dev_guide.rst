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

Installation for Development
----------------------------

Installation with Conda
```````````````````````

Perhaps the easiest way to install NiMARE for development is with Conda.

In this setup, you simply create a conda environment,
then install your local version of NiMARE in editable mode (``pip install -e``).

.. code-block:: bash

  cd /path/to/nimare_repo

  conda create -p /path/to/nimare_env pip python=3.10
  conda activate /path/to/nimare_env
  pip install -e .[all]

In this setup, any changes you make to your local clone of NiMARE will automatically be reflected in your environment.

Installation with Docker
````````````````````````

If you want more control over your installation (e.g., if you want non-Python packages installed as well),
using a Docker container may be the way to go.

Here are some common steps for taking this approach:

To build the Docker image:

.. code-block:: bash

  cd /path/to/nimare_repo
  docker build -t test/nimare .

To run the Docker container:

.. code-block:: bash

  docker run -it -v `pwd`:/home/neuro/code/NiMARE -p8888:8888 test/nimare bash

Once inside the container, you can install NiMARE:

.. code-block:: bash

  python /home/neuro/code/NiMARE/setup.py develop

Maintaining NiMARE
------------------

Labeling PRs
````````````

All PRs should be appropriately labeled.
PR labels determine how PRs will be reported in the next release's release notes.
For example, PRs with the "enhancement" label will be placed in the "🎉 Exciting New Features" section.

If you forget to add the appropriate labels to any PRs that you merge,
you can add them after they've been merged (and even change the titles),
as long as you do so before the next release has been published.

Making a Release
````````````````

To make a NiMARE release, use GitHub's online release tool.
Choose a new version tag, according to the semantic versioning standard.
The release title should be the same as the new tag (e.g., ``0.0.12``).
For pre-releases, we use release candidate terminology (e.g., ``0.0.12rc1``) and we select the "This is a pre-release" option.

At the top of the release notes, add some information summarizing the release.
After you have written the summary, use the "Generate release notes" button;
this will add the full list of changes in the new release based on our template.

Once the release notes have been completed, you can publish the release.
This will make the release on GitHub and will also trigger GitHub Actions to
(1) publish the new release to PyPi and (2) update the changelog file.

Updating the Changelog
``````````````````````

The Changelog is a Markdown file that is updated automatically by a GitHub Action.
It is a list of all releases, with a copy of the release notes for each release.
Since the Changelog is updated automatically, you should not have to edit it manually very often;
however, there are some cases where you may need to do so.
One common situation is a mistake in the release notes.
Another possibility is that someone identifies a bug in a past release,
in which case it may be useful to add an admonition to that release's release notes.

The auto-generated Changelog follows a specific format,
so it is important to understand its idiosyncrasies before making any manual changes.
The two main things to know are:

1.  All bullet points in the release notes **must** each be on a single line.
    You can't split up a list item across multiple lines, even though that is valid in both Markdown and restructedText.
2.  Admonitions must also be on a single line, which may look odd in restructuredText, but is necessary.
    Any admonitions that are formatted normally will be overwritten by the Action that updates the Changelog.

    Here is a typical admonition. It will not work with the Changelog Action.

    .. code-block:: text

      .. warning:: Known Bugs

          This version contains some bugs that were identified after it was released.

          - The ALESubtraction class from this release should not be used, as it uses a symmetric null distribution,
            which does not work properly for comparisons between Datasets with different sizes.

    Here is a valid single-line version of that admonition:

    .. code-block:: text

      .. warning:: Known Bugs This version contains some bugs that were identified after it was released.
      \   - The ALESubtraction class from this release should not be used, as it uses a symmetric null distribution, which does not work properly for comparisons between Datasets with different sizes.

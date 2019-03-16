.. include:: links.rst

Contributing to NiMARE
======================

This document explains contributing to ``NiMARE`` at a very high level,
with a focus on project governance and development philosophy.
For a more practical guide to the NiMARE development, please see our
`contributing guide`_.

.. _contributing guide: https://github.com/neurostuff/NiMARE/blob/master/CONTRIBUTING.md

.. _governance:

Governance
----------

Governance is a hugely important part of any project.
It is especially important to have clear process and communication channels
for open source projects that rely on a distributed network of volunteers, such as ``NiMARE``.

``NiMARE`` is currently supported by a small group of X core developers.
Even with only X members involved in decision making processes,
we've found that setting expectations and communicating a shared vision has great value.

By starting the governance structure early in our development,
we hope to welcome more people into the contributing team.
We are committed to continuing to update the governance structures as necessary.
Every member of the ``NiMARE`` community is encouraged to comment on these processes and suggest improvements.

As the first interim `Benevolent Dictator for Life (BDFL)`_,
Taylor Salo is ultimately responsible for any major decisions pertaining to ``NiMARE`` development.
However, all potential changes are explicitly and openly discussed in the described channels of
communication, and we strive for consensus amongst all community members.

.. _Benevolent Dictator for Life (BDFL): https://en.wikipedia.org/wiki/Benevolent_dictator_for_life

Code of conduct
```````````````

All ``NiMARE`` community members are expected to follow our `code of conduct`_
during any interaction with the project.
That includes---but is not limited to---online conversations,
in-person workshops or development sprints, and when giving talks about the software.

As stated in the code, severe or repeated violations by community members may result in exclusion
from collective decision-making and rejection of future contributions to the ``NiMARE`` project.

.. _code of conduct: https://github.com/neurostuff/NiMARE/blob/master/Code_of_Conduct.md

NiMARE's development philosophy
--------------------------------------

In contributing to any open source project,
we have found that it is hugely valuable to understand the core maintainers' development philosophy.
In order to aid other contributors in on-boarding to ``NiMARE`` development,
we have therefore laid out our shared opinion on several major decision points.
These are:

#. :ref:`exposing options to the user`,
#. :ref:`prioritizing project developments`,
#. :ref:`future-proofing for continuous development`, and
#. :ref:`when to release new software versions`


.. _exposing options to the user:

Which options are available to users?
`````````````````````````````````````

The ``NiMARE``  developers are committed to providing useful and interpretable outputs
for a majority of use cases.

The ``NiMARE`` "opinionated approach" is therefore to provide reasonable defaults and to hide some
options from the top level workflows.

This decision has two key benefits:

1. By default, users should get high quality results from running the pipelines, and
2. The work required of the ``NiMARE``  developers to maintain the project is more focused
   and somewhat restricted.

It is important to note that ``NiMARE``  is shipped under `an MIT license`_ which means that
the code can---at all times---be cloned and re-used by anyone for any purpose.

"Power users" will always be able to access and extend all of the options available.
We encourage those users to feed back their work into ``NiMARE``  development,
particularly if they have good evidence for updating the default values.

We understand that it is possible to build the software to provide more
options within the existing framework, but we have chosen to focus on `the 80 percent use cases`_.

You can provide feedback on this philosophy through any of the channels
listed on the ``NiMARE`` :doc:`support` page.

.. _an MIT license: https://github.com/neurostuff/NiMARE/blob/master/LICENSE
.. _the 80 percent use cases: https://en.wikipedia.org/wiki/Pareto_principle#In_software


.. _prioritizing project developments:

Structuring project developments
````````````````````````````````

The ``NiMARE``  developers have chosen to structure ongoing development around specific goals.
When implemented successfully, this focuses the direction of the project and helps new contributors
prioritize what work needs to be completed.

We have outlined our goals for ``NiMARE`` in our :doc:`roadmap`,
which we encourage all contributors to read and give feedback on.
Feedback can be provided through any of the channels listed on our :doc:`support` page.

In order to more directly map between our :doc:`roadmap` and ongoing `project issues`_,
we have also created `milestones in our github repository`_.

.. _project issues: https://github.com/neurostuff/NiMARE/issues
.. _milestones in our github repository: https://github.com/neurostuff/NiMARE/milestones

This allows us to:

1. Label individual issues as supporting specific aims, and
2. Measure progress towards each aim's concrete deliverable(s).


.. _future-proofing for continuous development:

How does ``NiMARE`` future-proof its development?
`````````````````````````````````````````````````

``NiMARE``  is a reasonably young project that is run by volunteers.
No one involved in the development is paid for their time.
In order to focus our limited time, we have made the decision to not let future possibilities limit
or over-complicate the most immediately required features.
That is, to `not let the perfect be the enemy of the good`_.

.. _not let the perfect be the enemy of the good: https://en.wikipedia.org/wiki/Perfect_is_the_enemy_of_good

While this stance will almost certainly yield ongoing refactoring as the scope of the software expands,
the team's commitment to transparency, reproducibility, and extensive testing
mean that this work should be relatively manageable.

We hope that the lessons we learn building something useful in the short term will be
applicable in the future as other needs arise.


.. _when to release new software versions:

When to release a new version
`````````````````````````````

In the broadest sense, we have adopted a "you know it when you see it" approach
to releasing new versions of the software.

To try to be more concrete, if a change to the project substantially changes the user's experience
of working with ``NiMARE``, we recommend releasing an updated version.
Additional functionality and bug fixes are very clear opportunities to release updated versions,
but there will be many other reasons to update the software as hosted on `PyPi`_.

.. _PyPi: https://pypi.org/project/NiMARE/

To give two concrete examples of slightly less obvious cases:

1. A substantial update to the documentation that makes ``NiMARE``  easier to use **would** count as
a substantial change to ``NiMARE``  and a new release should be considered.

2. In contrast, updating code coverage with additional unit tests does not affect the
**user's** experience with ``NiMARE``  and therefore does not require a new release.

Any member of the ``NiMARE``  community can propose that a new version is released.
They should do so by opening an issue recommending a new release and giving a
1-2 sentence explanation of why the changes are sufficient to update the version.
More information about what is required for a release to proceed is available
in the :ref:`release checklist`.


.. _release checklist:

Release Checklist
"""""""""""""""""

This is the checklist of items that must be completed when cutting a new release of NiMARE.
These steps can only be completed by a project maintainer, but they are a good resource for
releasing your own Python projects!

    #. All continuous integration must be passing and docs must be building successfully.

We have set up NiMARE so that releases automatically mint a new DOI with Zenodo;
a guide for doing this integration is available `here`_.

Release Process
"""""""""""""""

- We use `Release-drafter`_ to automatically generate the changelog.
- When it's time to release, copy the output of Release-drafter into CHANGES.rst, cleaning up any entries for clarity, grammar, etc., and sorting types of change into [FIX,ENH,DOC,RF,TEST,MAINT,CI], but otherwise maintaining order.
- Commit: ``git commit -m ‘[skip ci] REL: <version>’``
- Signed and annotated tag: ``git tag -s -a``
- Reformat the changelog entry into a release on GitHub (e.g. adding the Release Notes and CHANGES headers)
- Once the Zenodo entry is created, manually copy the DOI badge to the top of the GitHub release.


.. _`upload it to PyPi`: https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives
.. _`guide for creating a release on GitHub`: https://help.github.com/articles/creating-releases/
.. _`Release-drafter`: https://github.com/apps/release-drafter
.. _here: https://guides.github.com/activities/citable-code/

Getting Involved
----------------

If you find a bug or would like to add or improve features of NiMARE check out our `GitHub repository`_.

.. _GitHub repository: https://github.com/neurostuff/NiMARE/blob/master/CONTRIBUTING.md

If you have questions, or need help with using NiMARE check out `NeuroStars`_.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" NiMARE setup script """


def main():
    """ Install entry-point """
    import versioneer
    from io import open
    import os.path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages
    from nimare.info import (
        __packagename__,
        __version__,
        __author__,
        __email__,
        __maintainer__,
        __license__,
        __description__,
        __longdesc__,
        __url__,
        DOWNLOAD_URL,
        CLASSIFIERS,
        REQUIRES,
        TESTS_REQUIRES,
        EXTRA_REQUIRES,
        ENTRY_POINTS,
    )

    pkg_data = {
        'nimare': [
            'tests/data/*',
        ]
    }

    root_dir = op.dirname(op.abspath(getfile(currentframe())))

    version = None
    cmdclass = {}
    if op.isfile(op.join(root_dir, 'nimare', 'VERSION')):
        with open(op.join(root_dir, 'nimare', 'VERSION')) as vfile:
            version = vfile.readline().strip()
        pkg_data['nimare'].insert(0, 'VERSION')

    if version is None:
        version = versioneer.get_version()
        cmdclass = versioneer.get_cmdclass()

    setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        long_description=__longdesc__,
        author=__author__,
        author_email=__email__,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        url=__url__,
        license=__license__,
        classifiers=CLASSIFIERS,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        install_requires=REQUIRES,
        tests_require=TESTS_REQUIRES,
        extras_require=EXTRA_REQUIRES,
        entry_points=ENTRY_POINTS,
        packages=find_packages(exclude=("tests",)),
        package_data=pkg_data,
        zip_safe=False,
        cmdclass=cmdclass
    )


if __name__ == '__main__':
    main()

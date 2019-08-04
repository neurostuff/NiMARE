#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" NiMARE setup script """


def main():
    """ Install entry-point """
    import pprint
    import versioneer
    from io import open
    import os.path as op
    from inspect import getfile, currentframe
    from setuptools import setup, find_packages

    ver_file = op.join('nimare', 'info.py')
    with open(ver_file) as f:
        exec(f.read())
    vars = locals()

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
        name=vars['PACKAGENAME'],
        version=vars['VERSION'],
        description=vars['DESCRIPTION'],
        long_description=vars['LONGDESC'],
        author=vars['AUTHOR'],
        author_email=vars['EMAIL'],
        maintainer=vars['MAINTAINER'],
        maintainer_email=vars['EMAIL'],
        url=vars['URL'],
        license=vars['LICENSE'],
        classifiers=vars['CLASSIFIERS'],
        download_url=vars['DOWNLOAD_URL'],
        # Dependencies handling
        install_requires=vars['REQUIRES'],
        tests_require=vars['TESTS_REQUIRES'],
        extras_require=vars['EXTRA_REQUIRES'],
        entry_points=vars['ENTRY_POINTS'],
        packages=find_packages(exclude=("tests",)),
        package_data=pkg_data,
        zip_safe=False,
        cmdclass=cmdclass
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""NiMARE setup script."""
from setuptools import setup

import versioneer

if __name__ == "__main__":
    setup(
        name="NiMARE",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        zip_safe=False,
    )

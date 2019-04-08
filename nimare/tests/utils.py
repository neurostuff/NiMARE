"""
Utility functions for testing nimare.
"""
import os
import os.path as op
import shutil
import zipfile
from glob import glob

import requests


def get_test_data_path():
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), 'data') + op.sep)


def _download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    req = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f_obj:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f_obj.write(chunk)
    return local_filename


def download_nidm_pain(out_dir=None):
    """
    Download NIDM Results for 21 pain studies from NeuroVault for tests.
    """
    url = 'https://neurovault.org/collections/1425/download'
    if out_dir is None:
        out_dir = op.join(os.getcwd(), 'resources', 'data', 'neurovault-data',
                          'collection-1425')
        os.makedirs(out_dir, exist_ok=True)
    else:
        os.makedirs(out_dir, exist_ok=True)

    # Download
    fname = _download_file(url)

    # Unzip
    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    collection_folders = [f for f in glob(op.join(out_dir, '*'))
                          if '.nidm' not in f]
    collection_folders = [f for f in collection_folders if op.isdir(f)]
    if len(collection_folders) > 1:
        raise Exception('More than one folder found: '
                        '{0}'.format(', '.join(collection_folders)))
    else:
        folder = collection_folders[0]
    zip_files = glob(op.join(folder, '*.zip'))
    for zf in zip_files:
        fn = op.splitext(op.basename(zf))[0]
        with zipfile.ZipFile(zf, 'r') as zip_ref:
            zip_ref.extractall(op.join(out_dir, fn))

    os.remove(fname)
    shutil.rmtree(folder)

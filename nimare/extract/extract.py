import os
import os.path as op
import zipfile
from glob import glob
import shutil
import tarfile

from .utils import _get_dataset_dir, _download_zipped_file


def download_nidm_pain(data_dir=None, overwrite=False, verbose=1):
    """
    Download NIDM Results for 21 pain studies from NeuroVault for tests.

    Parameters
    ----------
    data_dir : :obj:`str`, optional
        Location in which to place the studies. Default is None, which uses the
        package's default path for downloaded data.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.
    verbose : :obj:`int`, optional
        Default is 1.

    Returns
    -------
    data_dir : :obj:`str`
        Updated data directory pointing to dataset files.
    """
    url = 'https://neurovault.org/collections/1425/download'

    dataset_name = "nidm_21pain"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if op.isdir(data_dir) and overwrite is False:
        return data_dir

    # Download
    fname = op.join(data_dir, url.split('/')[-1])
    _download_zipped_file(url, filename=fname)

    # Unzip
    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    collection_folders = [f for f in glob(op.join(data_dir, '*')) if '.nidm' not in f]
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
            zip_ref.extractall(op.join(data_dir, fn))

    os.remove(fname)
    shutil.rmtree(folder)
    return data_dir


def download_mallet(data_dir=None, overwrite=False, verbose=1):
    """
    Download the MALLET toolbox for LDA topic modeling.

    Parameters
    ----------
    data_dir : :obj:`str`, optional
        Location in which to place MALLET. Default is None, which uses the
        package's default path for downloaded data.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.
    verbose : :obj:`int`, optional
        Default is 1.

    Returns
    -------
    data_dir : :obj:`str`
        Updated data directory pointing to MALLET files.
    """
    url = 'https://neurovault.org/collections/1425/download'

    dataset_name = "mallet"

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    if op.isdir(data_dir) and overwrite is False:
        return data_dir

    mallet_file = op.join(data_dir, 'mallet-2.0.7.tar.gz')
    _download_zipped_file('http://mallet.cs.umass.edu/dist/mallet-2.0.7.tar.gz',
                          mallet_file)

    with tarfile.open(mallet_file) as tf:
        tf.extractall(path=data_dir)
    os.remove(mallet_file)
    return data_dir

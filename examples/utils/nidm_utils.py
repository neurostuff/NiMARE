"""
Utilities for reading NIDM Results packs.
"""
import os.path as op

import numpy as np

from nimare.utils import get_resource_path


def _get_file(cdict, t, data_dir):
    """
    Return the file associated with a given data type within a
    folder if it exists. Otherwise, returns an empty list.
    """
    temp = ''
    if t == 'con':
        temp = cdict['images'].get('con')
    elif t == 'se':
        temp = cdict['images'].get('se')
    elif t == 't':
        temp = cdict['images'].get('t')
    elif t == 'z':
        temp = cdict['images'].get('z')
    elif t == 't!z':
        # Get t-image only if z-image doesn't exist
        temp = cdict['images'].get('z')
        if temp is None:
            temp = cdict['images'].get('t')
        else:
            temp = None
    elif t == 'n':
        temp = cdict.get('sample_sizes', [])
        if temp:
            temp = np.mean(temp)
    else:
        raise Exception('Input type "{0}" not recognized.'.format(t))

    if isinstance(temp, str):
        temp = op.join(data_dir, temp)

    return temp


def get_files(ddict, types, data_dir=None):
    """
    Returns a list of files associated with a given data type
    from a set of subfolders within a directory. Allows for
    multiple data types and only returns a set of files from folders
    with all of the requested types.
    """
    if data_dir is None:
        data_dir = op.join(get_resource_path(), 'data')

    all_files = []
    for study in ddict.keys():
        files = []
        cdict = ddict[study]['contrasts']['1']
        for t in types:
            temp = _get_file(cdict, t, data_dir)
            if temp:
                files.append(temp)

        if len(files) == len(types):
            all_files.append(files)
    all_files = list(map(list, zip(*all_files)))
    return all_files

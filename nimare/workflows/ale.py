import os
import pathlib
from ..dataset.extract import convert_sleuth_to_database
from ..meta.cbma import ALE


def ale_sleuth_inference(sleuth_file, output_dir=None, output_prefix=None, n_iters=10000):
    dset = convert_sleuth_to_database(sleuth_file).get_dataset()
    ale = ALE(dset, ids=dset.ids)
    ale.fit(n_iters=n_iters, ids=dset.ids)

    if output_dir is None:
        output_dir = os.path.dirname(sleuth_file)
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        base = os.path.basename(sleuth_file)
        output_prefix, _ = os.path.splitext(base)
        output_prefix += '_'

    for name, img in ale.results.images.items():
        img.to_filename(os.path.join(output_dir, output_prefix + name + ".nii.gz"))




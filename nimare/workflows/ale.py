import nimare.dataset as ds
from nimare.meta.cbma import kernel
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
from scipy.io import savemat, loadmat
from os.path import join

from ..dataset.extract import convert_sleuth_to_database
from ..meta.cbma.ale import ALE
from ..due import due, Doi
import click

@click.command()
@click.argument('textfile', type=click.Path(exists=True, readable=True), help='Tab-delimited Sleuth text file to be ALE meta-analyzed.')
@click.argument('output_dir', type=click.Path(), help='Directory into which ALE results will be written.')
@click.argument('basename', type=click.Path(), help='Basename for written out ALE results.')

@due.dcite(Doi('10.1006/nimg.2002.1131'),
           description='Introduces activation likelihood estimation (ALE) for coordinate-based neuroimaging meta-analysis.')
@due.dcite(Doi('10.1002/hbm.20136'),
           description='Controlling the false discovery rate in activation likelihood estimation (ALE) meta-analysis.')
@due.dcite(Doi('10.1016/j.neuroimage.2016.04.072'),
           description='Update the recommendations for thresholding in activation likelihood estimation (ALE) meta-analsis.')

def sleuth_ale_workflow(text_file, kernel=ALEKernel, output_dir):
    db = convert_sleuth_to_database(text_file)
    dset = db.get_dataset(target='mni152_2mm')
    ale_obj = ALE(dset, kernel_estimator=kernel)
    ale_obj.fit(dset.ids, )
    ale_obj.results.save_results(output_dir=output_dir, prefix=basename)

if __name__ == '__main__':
    sleuth_ale_workflow()

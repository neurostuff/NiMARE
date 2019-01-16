from ..meta.cbma import kernel
from ..due import due, Doi

from nimare.dataset.extract import convert_sleuth_to_database
from nimare.meta.cbma.ale import ALE
from nimare.due import due, Doi
from sklearn import cluster
import pandas as pd
#from ..dataset.extract import convert_sleuth_to_database
#from ..meta.cbma.ale import ALE
#from ..due import due, Doi
import click

n_iters_default = 10000

@click.command(name='scale')
@click.argument('database', required=True, type=click.Path(exists=True, readable=True), help='NiMARE database or Sleuth text file containing meta-analytic data to be SCALEd.')
@click.argument('output_dir', required=True, type=click.Path(), help='Directory into which clustering results will be written.')
@click.argument('output_prefix', type=string, help='Common prefix for output SCALE results.')
@click.option('--n_iters', default=n_iters_default, show_default=True, type=int, help='Number of iterations for SCALE to perform in the likelihood estimation.')
@click.argument('base_img', type=click.Path(exists=True, readable=True), help='Voxelwise baseline activation rates.')

@due.dcite(Doi(''),
           description='Introduces Specific CoActivation Likelihood Estimation (SCALE) for meta-analytic coactivation modeling.')

def scale_workflow(database, output_dir, output_prefix, kernel_estimator, n_iters, base_img):
    #db = Database(database)
    #dset = db.get_dataset()
    #dataset from sleuth for now
    if database.endswith('.json'):
        db = database
    if database.endswith('.txt'):
        db = convert_sleuth_to_database(database)
    dset = db.get_dataset()
    ijk = np.loadtxt(base_img)
    estimator = SCALE(dset, ijk=ijk, kernel_estimator=ALEKernel, n_iters=n_iters)
    estimator.fit(dset.ids, voxel_thresh=0.001, n_iters=10000, n_cores=4)
    estimator.save_results(output_dir=output_dir, prefix=output_prefix, prefix_sep='_')
    print('Done! :)')

if __name__ == '__main__':
    scale_workflow()

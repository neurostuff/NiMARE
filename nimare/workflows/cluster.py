from ..meta.cbma.kernel import ALEKernel
from ..due import due, Doi


@due.dcite(Doi('10.1073/pnas.0905267106'),
           description='Introduces meta-analytic clustering analysis.')
def meta_cluster_workflow(dataset, n_clusters=[5]):
    pass

from ..meta.cbma.kernel import ALEKernel
from ..due import due, Doi


@due.dcite(Doi('10.1002/hbm.20854'),
           description='Introduces MACM.')
def macm_workflow(dataset, target_mask, kernel_estimator=ALEKernel, **kwargs):
    kernel_args = {k.split('kernel__')[1]: v for k, v in kwargs.items()
                   if k.startswith('kernel__')}
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith('kernel__')}

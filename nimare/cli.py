import click

from nimare.workflows.ale import ale_sleuth_workflow
from nimare.workflows.ibma_perm import con_perm_workflow
from nimare.workflows.macm import macm_workflow
# from nimare.workflows.cluster import meta_cluster_workflow
# from nimare.workflows.scale import scale_workflow
from nimare.workflows.peaks2maps import peaks2maps_workflow
from nimare.workflows.conversion import sleuth2nimare, neurosynth2nimare

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(ale_sleuth_workflow)
cli.add_command(con_perm_workflow)
cli.add_command(macm_workflow)
# cli.add_command(meta_cluster_workflow)
# cli.add_command(scale_workflow)
cli.add_command(peaks2maps_workflow)
cli.add_command(sleuth2nimare)
cli.add_command(neurosynth2nimare)

import click

from nimare.workflows.ale import ale_sleuth_inference
from nimare.workflows.cluster import meta_cluster_workflow
from nimare.workflows.scale import scale_workflow


@click.group()
def cli():
    pass


cli.add_command(ale_sleuth_inference)
cli.add_command(meta_cluster_workflow)
cli.add_command(scale_workflow)

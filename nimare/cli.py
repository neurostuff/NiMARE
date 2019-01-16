import click

from nimare.workflows.ale import ale_sleuth_inference
from nimare.workflows.peaks2maps import peaks2maps


@click.group()
def cli():
    pass


cli.add_command(ale_sleuth_inference)
cli.add_command(peaks2maps)

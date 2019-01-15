import click

from nimare.workflows.ale import ale_sleuth_inference


@click.group()
def cli():
    pass


cli.add_command(ale_sleuth_inference)

"""
Workflow for running an ALE meta-analysis from a Sleuth text file.
"""
import click

from ..io import convert_sleuth_to_json, convert_neurosynth_to_json


@click.command(name='sleuth2nimare',
               short_help='Convert a Sleuth text file to a NiMARE json file.')
@click.argument('sleuth_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path(exists=False))
def sleuth2nimare(sleuth_file, out_file):
    convert_sleuth_to_json(sleuth_file, out_file)


@click.command(name='neurosynth2nimare',
               short_help='Convert a Neurosynth text file to a NiMARE json '
                          'file.')
@click.argument('neurosynth_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path(exists=False))
@click.option('--annotations_file', type=click.Path(exists=True))
def neurosynth2nimare(neurosynth_file, out_file, annotations_file=None):
    convert_neurosynth_to_json(neurosynth_file, out_file, annotations_file)

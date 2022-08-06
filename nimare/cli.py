"""Command-line interfaces for common workflows."""
import argparse
import os.path as op

from nimare.io import convert_neurosynth_to_json, convert_sleuth_to_json
from nimare.workflows.ale import ale_sleuth_workflow
from nimare.workflows.macm import macm_workflow


def _is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error(f"The file {arg} does not exist!")

    return arg


def _get_parser():
    """Parse command line inputs for NiMARE.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(prog="nimare")
    subparsers = parser.add_subparsers(help="NiMARE workflows")
    # ALE sleuth workflow
    ale_parser = subparsers.add_parser(
        "ale",
        help=(
            "Run an activation likelihood estimation (ALE) meta-analysis on "
            "a Sleuth text file. ALE is a permutation-based meta-analysis "
            "of coordinates that uses 3D Gaussians to model activation."
        ),
    )
    ale_parser.set_defaults(func=ale_sleuth_workflow)
    ale_parser.add_argument(
        "sleuth_file",
        type=lambda x: _is_valid_file(parser, x),
        help=("Sleuth text file to analyze."),
    )
    ale_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    ale_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    ale_parser.add_argument(
        "--file2",
        dest="sleuth_file2",
        type=str,
        help=("Optional second Sleuth file for subtraction analysis."),
        default=None,
    )
    ale_parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        help=("Number of iterations for permutation testing."),
        default=10000,
    )
    ale_parser.add_argument(
        "--v_thr",
        dest="v_thr",
        type=float,
        help=("Voxel p-value threshold used to create clusters."),
        default=0.001,
    )
    ale_parser.add_argument(
        "--fwhm",
        dest="fwhm",
        type=float,
        help=(
            "Override sample size-based kernel determination with a "
            "single FWHM (in mm) applied to all experiments. Useful "
            "when sample size is not available for all data."
        ),
        default=None,
    )
    ale_parser.add_argument(
        "--n_cores",
        dest="n_cores",
        type=int,
        default=1,
        help=("Number of processes to use for meta-analysis. If -1, use all available cores."),
    )

    # MACM
    macm_parser = subparsers.add_parser(
        "macm",
        help=(
            "Run a meta-analytic coactivation modeling (MACM) "
            "analysis using activation likelihood estimation "
            "(ALE) on a NiMARE dataset file and a target mask."
        ),
    )
    macm_parser.set_defaults(func=macm_workflow)
    macm_parser.add_argument(
        "dataset_file", type=lambda x: _is_valid_file(parser, x), help=("Dataset file to analyze.")
    )
    macm_parser.add_argument(
        "--mask",
        "--mask_file",
        dest="mask_file",
        type=lambda x: _is_valid_file(parser, x),
        help=("Mask file"),
        required=True,
    )
    macm_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    macm_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    macm_parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        help=("Number of iterations for permutation testing."),
        default=10000,
    )
    macm_parser.add_argument(
        "--v_thr",
        dest="v_thr",
        type=float,
        help=("Voxel p-value threshold used to create clusters."),
        default=0.001,
    )
    macm_parser.add_argument(
        "--n_cores",
        dest="n_cores",
        type=int,
        default=1,
        help=("Number of processes to use for meta-analysis. If -1, use all available cores."),
    )

    # Conversion workflows
    sleuth2nimare_parser = subparsers.add_parser(
        "sleuth2nimare", help=("Convert a Sleuth text file to a NiMARE json file.")
    )
    sleuth2nimare_parser.set_defaults(func=convert_sleuth_to_json)
    sleuth2nimare_parser.add_argument(
        "text_file",
        type=lambda x: _is_valid_file(parser, x),
        help=("Sleuth text file to convert."),
    )
    sleuth2nimare_parser.add_argument("out_file", type=str, help=("Output file."))

    neurosynth2nimare_parser = subparsers.add_parser(
        "neurosynth2nimare", help=("Convert a Neurosynth text file to a NiMARE json file.")
    )
    neurosynth2nimare_parser.set_defaults(func=convert_neurosynth_to_json)
    neurosynth2nimare_parser.add_argument(
        "text_file",
        type=lambda x: _is_valid_file(parser, x),
        help=("Neurosynth text file to convert."),
    )
    neurosynth2nimare_parser.add_argument("out_file", type=str, help=("Output file."))
    neurosynth2nimare_parser.add_argument(
        "--annotations_file",
        metavar="FILE",
        type=lambda x: _is_valid_file(parser, x),
        help=("Optional annotations (features) file."),
        default=None,
    )

    return parser


def _main(argv=None):
    """Run NiMARE CLI entrypoint."""
    options = _get_parser().parse_args(argv)
    args = vars(options).copy()
    args.pop("func")
    options.func(**args)


if __name__ == "__main__":
    _main()

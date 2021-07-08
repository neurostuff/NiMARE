"""Command-line interfaces for common workflows."""
import argparse
import os.path as op

from nimare.io import convert_neurosynth_to_json, convert_sleuth_to_json
from nimare.workflows.ale import ale_sleuth_workflow
from nimare.workflows.cluster import meta_cluster_workflow
from nimare.workflows.conperm import conperm_workflow
from nimare.workflows.macm import macm_workflow
from nimare.workflows.peaks2maps import peaks2maps_workflow
from nimare.workflows.scale import scale_workflow


def _is_valid_file(parser, arg):
    """Check if argument is existing file."""
    if not op.isfile(arg) and arg is not None:
        parser.error("The file {0} does not exist!".format(arg))

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

    # Contrast permutation workflow
    conperm_parser = subparsers.add_parser(
        "conperm",
        help=(
            "Meta-analysis of contrast maps using random effects and "
            "two-sided inference with empirical (permutation-based) null "
            "distribution and Family Wise Error multiple comparisons "
            "correction. Input may be a list of 3D files or a single 4D "
            "file."
        ),
    )
    conperm_parser.set_defaults(func=conperm_workflow)
    conperm_parser.add_argument(
        "contrast_images",
        nargs="+",
        metavar="FILE",
        type=lambda x: _is_valid_file(parser, x),
        help=("Data to analyze. May be a single 4D file or a list of 3D files."),
    )
    conperm_parser.add_argument(
        "--mask",
        dest="mask_image",
        metavar="FILE",
        type=lambda x: _is_valid_file(parser, x),
        help=("Mask file."),
        default=None,
    )
    conperm_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    conperm_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    conperm_parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        help=("Number of iterations for permutation testing."),
        default=10000,
    )

    # Contrast permutation applied to Peaks2Maps-reconstructed maps
    peaks2maps_parser = subparsers.add_parser(
        "peaks2maps",
        help=(
            "Method for performing coordinate-based meta-analysis that "
            "uses a pretrained deep neural network to reconstruct "
            "unthresholded maps from peak coordinates. The reconstructed "
            "maps are evaluated for statistical significance using a "
            "permutation-based approach with Family Wise Error multiple "
            "comparison correction."
        ),
    )
    peaks2maps_parser.set_defaults(func=peaks2maps_workflow)
    peaks2maps_parser.add_argument(
        "sleuth_file",
        type=lambda x: _is_valid_file(parser, x),
        help=("Sleuth text file to analyze."),
    )
    peaks2maps_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    peaks2maps_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    peaks2maps_parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        help=("Number of iterations for permutation testing."),
        default=10000,
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

    # SCALE
    scale_parser = subparsers.add_parser(
        "scale",
        help=(
            "Method for performing Specific CoActivation Likelihood "
            "Estimation (SCALE), a modified meta-analytic coactivation "
            "modeling (MACM) that takes activation frequency bias into "
            "account, for delineating distinct core networks of "
            "coactivation, using a permutation-based approach."
        ),
    )
    scale_parser.set_defaults(func=scale_workflow)
    scale_parser.add_argument(
        "dataset_file", type=lambda x: _is_valid_file(parser, x), help=("Dataset file to analyze.")
    )
    scale_parser.add_argument(
        "--baseline",
        type=lambda x: _is_valid_file(parser, x),
        help=("Voxel-wise baseline activation rates."),
    )
    scale_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    scale_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    scale_parser.add_argument(
        "--n_iters",
        dest="n_iters",
        type=int,
        help=("Number of iterations for permutation testing."),
        default=2500,
    )
    scale_parser.add_argument(
        "--v_thr",
        dest="v_thr",
        type=float,
        help=("Voxel p-value threshold used to create clusters."),
        default=0.001,
    )
    scale_parser.add_argument(
        "--n_cores",
        dest="n_cores",
        type=int,
        default=1,
        help=("Number of processes to use for meta-analysis. If -1, use all available cores."),
    )

    # Meta-analytic clustering
    cluster_parser = subparsers.add_parser(
        "metacluster",
        help=(
            "Method for investigating recurrent patterns of activation "
            "across a meta-analytic dataset, thus identifying trends across "
            "a collection of experiments."
        ),
    )
    cluster_parser.set_defaults(func=meta_cluster_workflow)
    cluster_parser.add_argument(
        "dataset_file", type=lambda x: _is_valid_file(parser, x), help=("Dataset file to analyze.")
    )
    cluster_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        metavar="PATH",
        type=str,
        help=("Output directory."),
        default=".",
    )
    cluster_parser.add_argument(
        "--prefix", dest="prefix", type=str, help=("Common prefix for output maps."), default=""
    )
    cluster_parser.add_argument(
        "--kernel",
        dest="kernel",
        choices=["ALEKernel", "MKDAKernel", "KDAKernel", "Peaks2MapsKernel"],
        help=("Kernel estimator, for coordinate-based metaclustering."),
        default="ALEKernel",
    )
    cluster_parser.add_argument(
        "--algorithm",
        dest="algorithm",
        choices=["kmeans", "dbscan", "spectral"],
        help=("Clustering algorithm to be used, from sklearn.cluster."),
        default="kmeans",
    )
    cluster_parser.add_argument(
        "--clust_range",
        dest="clust_range",
        type=int,
        nargs=2,
        help=(
            "Select a range for k over which clustering solutions will be "
            "evaluated (e.g., 2 10 will evaluate solutions with k = 2 "
            "clusters to k = 10 clusters)."
        ),
        required=True,
    )
    type_group = cluster_parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument(
        "--coord",
        dest="coord",
        action="store_true",
        help=("Is input data coordinate-based?"),
        default=False,
    )
    type_group.add_argument(
        "--img",
        dest="coord",
        action="store_false",
        help=("Is input data image-based?"),
        default=False,
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

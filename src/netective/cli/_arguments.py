__all__ = ["_parse_arguments"]

import os
import argparse

from netective.utils import concat_path


def _parse_arguments():

    parser = argparse.ArgumentParser(
        description="Assess the topology of a network. If more than one network is given (directory with multiple networks), a comparison between them based on their topology is done."
    )

    # Arguments for network analysis
        # Networks
        # Norm
    parser.add_argument(
        "--norm",
        metavar= "normalization",
        default= None,
        help="whether to normalize structural properties, default is None.",
        choices=[None, 'network', 'biological']
    )
        # Selected props
    parser.add_argument(
        "--props",
        metavar="selected_props",
        type=list,
        default=['all'],
        help="list of selected properties used for analysis, defaults to ['all'] (meaning all properties)",
    )
        # Workers
    parser.add_argument(
        "--workers",
        metavar="workers",
        type=str,
        default='1',
        help="number of workers to use, default is 1",
        choices=['auto', '1','int representing number of workers']
    )
        # Return props dict
    parser.add_argument(
        "--keep_props",
        metavar="keep properties",
        type=bool,
        default=False,
        help="whether to save dataframes of the properties values for each network analyzed, default is False"
    )
        # Verbose
    parser.add_argument(
        "--verbose",
        metavar="verbose",
        type=str,
        default='CRITICAL',
        help="level of verbose to handle progress of process. Check logging levels for more information. Defaults to CRITICAL",
        choices=['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL']
    )
        # Erdos Renyi
    parser.add_argument(
        "--erdos_renyi",
        metavar="erdos_renyi",
        type=int,
        default=10,
        help="number of Erdos-Renyi networks to generate for each network, default is 10",
    )

    # Technical arguments
        # Comments character in networks files
    parser.add_argument(
        "--comments",
        metavar="comments",
        type=str,
        default="#",
        help="character used to indicate comments in the network file(s)",
    )
        # Delimiter to parse networks
    parser.add_argument(
        "--delimiter",
        metavar="delimiter",
        type=str,
        default="\t",
        help="character used to separate columns in the network file(s)",
    )
        # Path to dump outputs
    parser.add_argument(
        "--output",
        metavar="output",
        type=str,
        default=os.getcwd(),
        help="path to output directory, default is current directory",
    )
        # Name for ouputs
    parser.add_argument(
        "--output_file",
        metavar="output_file",
        type=str,
        default="structural_properties",
        help="name of output file, default is structural_properties",
    )

    # TODO: https://stackoverflow.com/questions/52403065/argparse-optional-boolean implement
    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        "--path",
        metavar="path",
        type=str,
        help="path to network file or a folder containing network files",
        required=True,
    )

    ## parse arguments
    args = parser.parse_args()
    args.delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")
    """
    # valid file paths
    if os.path.isdir(args.path):
        file_paths = [
            concat_path(args.path, f)
            for f in os.listdir(args.path)
            if os.path.isfile((concat_path(args.path, f)))
        ]
    else:

        # raise error if file does not exist
        if not os.path.isfile(args.path):
            raise FileNotFoundError(f"File {args.path} does not exist.")

        file_paths = [args.path]
    args.nets_path = file_paths
    """
    ## parse arguments
    args = parser.parse_args()
    # args.delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")

    # valid output path
    if not os.path.isdir(args.output):
        raise NotADirectoryError(
            f"Output path {args.output} is not a valid directory."
        )
    args.output = os.path.abspath(args.output)

    return args

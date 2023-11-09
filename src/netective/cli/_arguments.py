__all__ = ["_parse_arguments"]

import os
import argparse

def _parse_arguments():

    def list_of_strings(arg):
        return [x.strip() for x in arg.split(',')]
    
    
    parser = argparse.ArgumentParser(
        description="Assess the topology of a network. If more than one network is given (directory with multiple networks), a comparison between them based on their topology is done."
    )

    # Arguments for network analysis
        # Norm
    parser.add_argument(
        '-n','--normalization',
        default= None,
        help= "normalization method for structural properties, default is no normalization.",
        choices= ['network', 'biological'],
        required= False
    )
        # Selected props
    parser.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
        default= ['all'],
        help= 'list of selected properties used for analysis, defaults to all properties implemented. Format accepted: "Gini Index, Density, Average Out-Degree of Nearest Neighbors, etc..."',
        required= False
    )
        # Workers
    parser.add_argument(
        '-w','--workers',
        type= int,
        default= None,
        help= "number of workers to use for parallelization of properties computation during network comparison, default is automatical detection of usable threads. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment.",
        required= False
    )
        # Return props dict
    parser.add_argument(
        '-k', '--keep_props',
        action= 'store_true',
        help= "whether to save dataframes of the properties values for each network analyzed",
        required= False
    )
        # Verbose
    parser.add_argument(
        '-v','--verbose',
        type= str,
        default= 'CRITICAL',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to CRITICAL",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Erdos Renyi
    parser.add_argument(
        '-er','--erdos_renyi',
        type= int,
        default= 0,
        help= "number of Erdos-Renyi networks to generate for each inputed network, default is 0",
        required= False
    )

    # Technical arguments
        # Comments character in networks files
    parser.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network file(s)",
        required= False
    )
        # Delimiter to parse networks
    parser.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network file(s)",
        required= False
    )
        # Path to dump outputs
    parser.add_argument(
        '-o','--output',
        type= str,
        default= os.getcwd(),
        help= "path to output directory, default is current directory",
        required= False
    )

    requiredNamed = parser.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= str,
        help= "path to network file or a folder containing network files",
        required= True,
    )

    ## parse arguments
    args = parser.parse_args()

    args.delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")

    # valid output path
    if not os.path.isdir(args.output):
        raise NotADirectoryError(
            f"Output path {args.output} is not a valid directory."
        )
    args.output = os.path.abspath(args.output)

    return args

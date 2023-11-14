__all__ = ['_parse_arguments']
import os
import argparse

def _parse_arguments():

    def list_of_strings(arg):
        return [x.strip() for x in arg.split(',')]
    
    
    parser = argparse.ArgumentParser(
        description="Assess the topology of a network. If more than one network is given (directory with multiple networks), a comparison between them based on their topology is done."
    )
    # Sub-Arguments
    subparsers = parser.add_subparsers(
        title='Runmodes',
        description='Valid runmodes',
        required=True
    )
    
    # create the parser for the "Characterize" command
    parser_a = subparsers.add_parser('Characterize', help='characterization of inputed network through structural properties.')
    parser_a.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 1
    )
        # Norm
    parser_a.add_argument(
        '-n','--normalization',
        default= None,
        help= "normalization method for structural properties, default is no normalization.",
        choices= ['network', 'biological'],
        required= False
    )
        # Selected props
    parser_a.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
        default= ['all'],
        help= 'list of selected properties used for analysis, defaults to all properties implemented. Format accepted: "Gini Index, Density, Average Out-Degree for Nearest Neighbors, etc..."',
        required= False
    )
        # Workers
    parser_a.add_argument(
        '-w','--workers',
        type= int,
        default= None,
        help= "number of workers to use for parallelization of properties characterization, default is automatical detection of usable threads. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. Only applies if directory has more than one readable network.",
        required= False
    )
        # Return props dict
    parser_a.add_argument(
        '-k', '--keep_props',
        action= 'store_true',
        help= "whether to save dataframes of the properties values for each network analyzed",
        required= False
    )
        # Verbose
    parser_a.add_argument(
        '-v','--verbose',
        type= str,
        default= 'CRITICAL',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to CRITICAL",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Comments character in networks files
    parser_a.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network file(s)",
        required= False
    )
        # Delimiter to parse networks
    parser_a.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network file(s)",
        required= False
    )
        # Path to dump outputs
    parser_a.add_argument(
        '-o','--output',
        type= str,
        default= os.getcwd(),
        help= "path to output directory, default is current directory",
        required= False
    )
    requiredNamed = parser_a.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= str,
        help= "path to folder containing network file(s).",
        required= True,
    )


    # create the parser for the "Compare" command
    parser_b = subparsers.add_parser('Compare', help='comparison between multiple networks based on their topologies.')
    parser_b.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 2
    )
        # Norm
    parser_b.add_argument(
        '-n','--normalization',
        default= None,
        help= "normalization method for structural properties, default is no normalization.",
        choices= ['network', 'biological'],
        required= False
    )
        # Selected props
    parser_b.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
        default= ['all'],
        help= 'list of selected properties used for analysis, defaults to all properties implemented. Format accepted: "Gini Index, Density, Average Out-Degree for Nearest Neighbors, etc..."',
        required= False
    )
        # Workers
    parser_b.add_argument(
        '-w','--workers',
        type= int,
        default= None,
        help= "number of workers to use for parallelization of properties computation during network comparison, default is automatical detection of usable threads. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment.",
        required= False
    )
        # Return props dict
    parser_b.add_argument(
        '-k', '--keep_props',
        action= 'store_true',
        help= "whether to save dataframes of the properties values for each network analyzed",
        required= False
    )
        # Verbose
    parser_b.add_argument(
        '-v','--verbose',
        type= str,
        default= 'CRITICAL',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to CRITICAL",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Erdos Renyi
    parser_b.add_argument(
        '-er','--erdos_renyi',
        type= int,
        default= 0,
        help= "number of Erdos-Renyi networks to generate for each inputed network, default is 0",
        required= False
    )
        # Comments character in networks files
    parser_b.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files",
        required= False
    )
        # Delimiter to parse networks
    parser_b.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network files",
        required= False
    )
        # Path to dump outputs
    parser_b.add_argument(
        '-o','--output',
        type= str,
        default= os.getcwd(),
        help= "path to output directory, default is current directory",
        required= False
    )
    requiredNamed = parser_b.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= str,
        help= "path to folder containing network files.",
        required= True,
    )

    # create the parser for the "Assess" command
    parser_c = subparsers.add_parser('Assess', help='statistical evaluation of inferences based on a Gold Standard.')
    parser_c.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 3
    )

    # create the parser for the "Classify" command
    parser_d = subparsers.add_parser('Classify', help='classification of networks into clusters and evaluation of said classification.')
    parser_d.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 4
    )
        # Norm
    parser_d.add_argument(
        '-n','--normalization',
        default= None,
        help= "normalization method for structural properties, default is no normalization.",
        choices= ['network', 'biological'],
        required= False
    )
        # Selected props
    parser_d.add_argument(
        '-p', '--selected_props',
        type= list,
        default= ['Average Local Efficiency',
            'Radius',
            'Center',
            'Periphery',
            'Complex Feed-Forward Circuits',
            'Feed-Forward Circuits',
            'Max Degree',
            'Gini Index',
            'Global Efficiency',
            'Undirected Gini Index',
            'Entropy of Degree Distribution',
            'Self-Loops'],
        help= argparse.SUPPRESS
    )
        # Workers
    parser_d.add_argument(
        '-w','--workers',
        type= int,
        default= None,
        help= "number of workers to use for parallelization of properties computation during network comparison, default is automatical detection of usable threads. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment.",
        required= False
    )
        # Return props dict
    parser_d.add_argument(
        '-k', '--keep_props',
        help= argparse.SUPPRESS,
        default= True
    )
        # Verbose
    parser_d.add_argument(
        '-v','--verbose',
        type= str,
        default= 'CRITICAL',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to CRITICAL",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Erdos Renyi
    parser_d.add_argument(
        '-er','--erdos_renyi',
        type= int,
        default= 0,
        help= "number of Erdos-Renyi networks to generate for each inputed network, default is 0",
        required= False
    )
        # Comments character in networks files
    parser_d.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files",
        required= False
    )
        # Delimiter to parse networks
    parser_d.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network files",
        required= False
    )
        # Path to dump outputs
    parser_d.add_argument(
        '-o','--output',
        type= str,
        default= os.getcwd(),
        help= "path to output directory, default is current directory",
        required= False
    )
    requiredNamed = parser_d.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= str,
        help= "path to folder containing network files.",
        required= True,
    )
    
    args = parser.parse_args()
    args.delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")
    # valid output path
    if not os.path.isdir(args.output):
        raise NotADirectoryError(
            f"Output path {args.output} is not a valid directory."
        )
    args.output = os.path.abspath(args.output)

    return args
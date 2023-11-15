__all__ = ['_parse_arguments']
import os
import argparse

def _parse_arguments():

    def list_of_strings(arg):
        return [x.strip() for x in arg.split(',')]
    
    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x
    
    
    parser = argparse.ArgumentParser(
        description="Assess the topology of a network. If more than one network is given (directory with multiple networks), a comparison between them based on their topology is done."
    )
    # Sub-Arguments
    subparsers = parser.add_subparsers(
        title='Runmodes',
        description='Valid runmodes',
        required=True
    )
    
    # create the parser for the "characterize" command
    parser_a = subparsers.add_parser('characterize', help='characterization of inputed network through structural properties.')
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
        type= str,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is 2 for minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network.",
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
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to WARNING",
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

    # create the parser for the "compare" command
    parser_b = subparsers.add_parser('compare', help='comparison between multiple networks based on their topologies.')
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
        type= str,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is 2 for minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network.",
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
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to WARNING.",
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

    # create the parser for the "assess" command
    parser_c = subparsers.add_parser('assess', help='statistical evaluation of inferences based on a Gold Standard.')
    parser_c.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 3
    )
    requiredNamed = parser_c.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-gs','--gold_standard',
        type= str,
        help= "path to gold standard network.",
        required= True,
    )
    requiredNamed.add_argument(
        '-inf','--inferences',
        type= str,
        help= "path to directory containing inferenced networks.",
        required= True,
    )
    parser_c.add_argument(
        '-g', '--greater_is_better',
        action= 'store_true',
        help= 'whether the inference score is better when it is higher or lower. If greater_is_better is not declared, the lower the score the better the inference.',
        required= False
    )
    parser_c.add_argument(
        '-sl', '--self_loops',
        action= 'store_true',
        help= 'whether the self-loops are allowed or not.',
        required= False
    )
    parser_c.add_argument(
        '-coff','--cutoff',
        default= False,
        type= restricted_float,
        help= "t distance for clustering, threshold to apply when forming flat clusters. IMPORTANT: if a threshold is given, no max number of clusters can apply.",
        required= False
    )
    parser_c.add_argument(
        '-dir', '--directed',
        action= 'store_true',
        help= 'whether the gold standard and inferences are directed or not.',
        required= False
    )
    parser_c.add_argument(
        '-k', '--keep_auc_dicts',
        action= 'store_true',
        help= 'whether to return the AUC values for every inference in the benchmark.',
        required= False
    )
    parser_c.add_argument(
        '-s', '--score',
        action= 'store_true',
        help= 'whether the inference networks have a score attribute or not.',
        required= False
    )
    parser_c.add_argument(
        '-v','--verbose',
        type= str,
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to WARNING.",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
    parser_c.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files",
        required= False
    )
    parser_c.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network files",
        required= False
    )
    parser_c.add_argument(
        '-o','--output',
        type= str,
        default= None,
        help= "path to output directory, default for results printed to std.out",
        required= False
    )

    # create the parser for the "classify" command
    parser_d = subparsers.add_parser('classify', help='classification of networks into clusters and evaluation of said classification.')
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
        # Mutually exclusive arguments
    mut_exclusive = parser_d.add_mutually_exclusive_group(required=False)
        # Clusters
    mut_exclusive.add_argument(
        '-cl','--clusters',
        default= None,
        type= int,
        help= "max number of clusters to classify networks into. IMPORTANT: if number of clusters is given, then no t distance can be given.",
        required= False
    )
        # Threshold distance
    mut_exclusive.add_argument(
        '-t','--threshold',
        default= 0.7,
        type= restricted_float,
        help= "t distance for clustering, threshold to apply when forming flat clusters. IMPORTANT: if a threshold is given, no max number of clusters can apply.",
        required= False
    )
        # Clustering metric
    parser_d.add_argument(
        '-m','--method',
        default= 'ward',
        type= str,
        help= "method for calculating the distance between the newly formed cluster v and each cluster u. Defaults to ward. See scipy.clusters.hierarchy.linkage for more info.",
        choices= ['single', 
                  'complete', 
                  'average', 
                  'weighted', 
                  'centroid', 
                  'median', 
                  'ward'],
        required= False
    )
        # Clustering method
    parser_d.add_argument(
        '-mtr','--metric',
        default= 'euclidean',
        type= str,
        help= 'distance metric to use. Defaults to euclidean.',
        choices= ['braycurtis', 
                  'canberra', 
                  'chebyshev', 
                  'cityblock', 
                  'correlation', 
                  'cosine', 
                  'dice', 
                  'euclidean', 
                  'hamming', 
                  'jaccard', 
                  'jensenshannon', 
                  'kulczynski1', 
                  'mahalanobis', 
                  'matching', 
                  'minkowski', 
                  'rogerstanimoto', 
                  'russellrao', 
                  'seuclidean', 
                  'sokalmichener', 
                  'sokalsneath', 
                  'sqeuclidean',
                  'yule'],
        required= False
    )
        # Selected props
    parser_d.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
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
        required= False,
        help= 'list of selected properties used for analysis, defaults to selected properties for best classification. Format accepted: "Gini Index, Density, Average Out-Degree for Nearest Neighbors, etc..."',
    )
        # Workers
    parser_d.add_argument(
        '-w','--workers',
        type= str,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is 2 for minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network.",
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
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information. Defaults to WARNING",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Erdos Renyi
    parser_d.add_argument(
        '-er','--erdos_renyi',
        type= int,
        default= 0,
        help= argparse.SUPPRESS
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
        default= None,
        help= "path to output directory, default for results printed to std.out",
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
    if args.output is not None:
        if not os.path.isdir(args.output):
            raise NotADirectoryError(
                f"Output path {args.output} is not a valid directory."
            )
        args.output = os.path.abspath(args.output)

    return args
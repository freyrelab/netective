__all__ = ['_parse_arguments']
import os
import argparse
from multiprocessing import cpu_count
import textwrap

def _parse_arguments():

    def list_of_strings(arg):
        return [x.strip() for x in arg.split(',')]
    
    def mixed_list(arg):
        return [x.strip() if not x.strip().isdigit() else int(x.strip()) for x in arg.split(',')]

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x
    
    def valid_path(x):
        if x is not None:
            if os.path.isdir(x) or os.path.isfile(x):
                return os.path.abspath(x)
            else:
                raise argparse.ArgumentTypeError(f'Output path {x} is not a valid directory')
        else: 
            return None
    
    def valid_workers(x):
        usable_threads = cpu_count() - 1
        try:
            x = int(x)
            x = usable_threads if x > usable_threads or x < 0 else x
        except:
            x = 'auto'
        return x

    parser = argparse.ArgumentParser(
        description="""\
        ( ◣_◢ )
        Welcome to Netective!
        Computational tool for network similarity analysis. Netective is capable of performing
        a comprehensive structural characterization of unweighted graphs from all domains, directed and/or undirected.
        Netective can then compare an unlimited number of networks through said structural characterization, as well as
        classifying them into clusters using two possible criterions.
        Netective can also perform an optimized statistical benchmarking of network inferences based on a Gold Standard.

        Netective was developed and is maintained by the team at FreyreLab, Center for Genomic Sciences, Autonomous University of Mexico.     
                                                -      -.
                                             -###      -###-
                                          +######      -######-
                                      .##########      -#########+
                                   .#############      -#############.
                                -################      -################.
                               ##################      -##################
                               ##################      -##################
                      .#-      ##################      -##################
                   -####-      ##################      -##################
                -#######-      ##################      -##################
             ###########-      ##################      -##################
          ##############-      ###############+.         -################
      .#################-      ############+                .#############
      ##################-      #########+                       +#########
      ##################-      ######-           .#####            #######
      ##################-      ###.           -###########.           +###
      ##################-      .           +#################-           .
      ##################-               ########################+
      ##################.            ##############################+
      ###############-            +###################################+
      ############.                  ##############################+
      #########            +            +#######################-
      #####+            +#####+.           -#################.           .
      ##-            +###########+.           .###########.           -###
                 -+##################-           .#####            +######
               .------------------------                        .--------- 
                                                                                     
                         -#                     .#   .#-
      .  ...     .--.   -##-   .---      ---.  .##-.  .  ..     .   .--.
      ##.  ##   #+  -#   ##   #+  .#-  +#.  +-  +#.  .#+  #-  .#-  ##  .#.
      #+   +#  ###+++#-  ##  .##+++## .##       -#   .#+  .#. +#  +##+++#+
      #+   +#  -#-       ##   #+       ##   .-  -#   .#+   -###   .#+
      #+   +#   .####-   +##-  +#+#+    +###+   .##+ .#+    ##.    .+#+#-                                                                                                                                                              
        """,
        formatter_class= argparse.RawDescriptionHelpFormatter
    )
    # Sub-Arguments
    subparsers = parser.add_subparsers(
        title='Runmodes',
        description="""\
        Netective has four main runmodes, each with different and specific arguments.
        Please consult each runmode's help manual before running any of them.
        Enjoy Netective!
        """,
        required=True
    )
    
    #########################################################################################################################
    # create the parser for the "characterize" command
    parser_a = subparsers.add_parser('characterize', help='characterization of inputed network/s through structural properties.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        help= "normalization method for structural properties.",
        choices= ['network', 'biological'],
        required= False
    )
    parser_a.add_argument(
        '-dir', '--directed',
        action= 'store_true',
        help= 'whether input networks are directed or not.',
        required= False
    )
        # Selected props
    parser_a.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
        default= ['all'],
        help= 'list of selected properties used for analysis. Accepted format: coma-separated string, written between "s.',
        required= False
    )
        # Workers
    parser_a.add_argument(
        '-w','--workers',
        type= valid_workers,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network.",
        required= False
    )
        # Return props dict
    parser_a.add_argument(
        '-k', '--keep_props',
        action= 'store_true',
        help= "whether to save dataframes of the properties values for each network analyzed.",
        required= False
    )
        # Verbose
    parser_a.add_argument(
        '-v','--verbose',
        type= str,
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information.",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Nets files format
    parser_a.add_argument(
        '-nff', '--net_f_format',
        type= str,
        default= 'edgelist',
        help= 'nets files format to parse.',
        choices= ['edgelist', 'graphml', 'adj list', 'multiline adj list'],
        required= False
    )

        # Comments character in networks files
    parser_a.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network file(s).",
        required= False
    )
        # Delimiter to parse networks
    parser_a.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network file(s).",
        required= False
    )
        # Path to dump outputs
    parser_a.add_argument(
        '-o','--output',
        type= valid_path,
        default= os.getcwd(),
        help= "path to output directory.",
        required= False
    )
    requiredNamed = parser_a.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= valid_path,
        help= "path to folder containing network file(s).",
        required= True,
    )

    ##########################################################################################
    # create the parser for the "compare" command
    parser_b = subparsers.add_parser('compare', help='comparison between multiple networks based on their topologies.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
        help= "normalization method for structural properties.",
        choices= ['network', 'biological'],
        required= False
    )
    parser_b.add_argument(
        '-dir', '--directed',
        action= 'store_true',
        help= 'whether input networks are directed or not.',
        required= False
    )
        # Selected props
    parser_b.add_argument(
        '-p', '--selected_props',
        type= list_of_strings,
        default= ['all'],
        help= 'list of selected properties used for analysis. Accepted format: coma-separated string, written between "s.',
        required= False
    )
        # Workers
    parser_b.add_argument(
        '-w','--workers',
        type= valid_workers,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network.",
        required= False
    )
        # Return props dict
    parser_b.add_argument(
        '-k', '--keep_props',
        action= 'store_true',
        help= "whether to save dataframes of the properties values for each network analyzed.",
        required= False
    )

        # Association metric
    parser_b.add_argument(
        '-a', '--association',
        type= str,
        default= 'pearson',
        help= 'association metric for calculating correlation between scalar properties arrays.',
        choices= ['pearson', 'spearman', 'cosine'],
        required= False
    )
    # Distance metric
    parser_b.add_argument(
        '-mtr','--metric',
        default= 'euclidean',
        metavar= 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist',
        type= str,
        help= 'distance metric to use. See scipy.spatial.distance.pdist() for more info.',
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
        # Linkage method
    parser_b.add_argument(
        '-m','--method',
        default= 'ward',
        type= str,
        metavar= 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html',
        help= "linkage method to use for calculating clusters. See scipy.clusters.hierarchy.linkage() for more info.",
        choices= ['single', 
                  'complete', 
                  'average', 
                  'weighted', 
                  'centroid', 
                  'median', 
                  'ward'],
        required= False
    )
    parser_b.add_argument(
        '-im', '--incmodels',
        type= list_of_strings,
        default= None,
        help= 'which random network generators to use to create analogs to each input network for comparison to them. Accepted format: coma-separated string, written between "s. Admitted models are: Erdos GNP, Erdos GNM, K Regular and Barabasi Albert.',
        required= False
    )
        # Compare to analog random models
    parser_b.add_argument(
        '-c2m', '--comp2models',
        action= 'store_true',
        help= 'whether to create a comparison heatmap of input networks to model analogs. Default behavior is to create symmetric heatmap of all analyzed networks (input and analogs).',
        required= False
    )
        # Number of analog random models
    parser_b.add_argument(
        '-nm', '--n_models',
        default = 2,
        type= int,
        help= 'number of analog networks from each random model to create for each input network.',
        required= False
    )
        # Direction for random analog models
    parser_b.add_argument(
        '-dirm', '--directed_models',
        action= 'store_true',
        help= 'whether analog random models will be created using direction, if possible.',
        required= False
    )

        # m to use in Barabasi Albert algorithm
    parser_b.add_argument(
        '-ba', '--m4ba',
        default= [2],
        type= mixed_list,
        help= 'm to use in Barbasi Albert algorithm. It can either be positive integers or degree distributions from each input network. Accepted format: coma-separated string, written between "s. Admitted distributions: in degree, out degree and undirected degree. String may include several positive integers aswell as any or all available distributions.',
        required= False
    )
    
        # Verbose
    parser_b.add_argument(
        '-v','--verbose',
        default= 'WARNING',
        type= str,
        help= "level of verbose to handle progress of process. Check logging levels for more information.",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
    
        # Nets files format
    parser_b.add_argument(
        '-nff', '--net_f_format',
        type= str,
        default= 'edgelist',
        help= 'nets files format to parse.',
        choices= ['edgelist', 'graphml', 'adj list', 'multiline adj list'],
        required= False
    )

        # Comments character in networks files
    parser_b.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files.",
        required= False
    )
        # Delimiter to parse networks
    parser_b.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network files.",
        required= False
    )
        # Path to dump outputs
    parser_b.add_argument(
        '-o','--output',
        type= valid_path,
        default= os.getcwd(),
        help= "path to output directory.",
        required= False
    )
        # Whether to keep distribution averages in scalars arrays
    parser_b.add_argument(
        '-noa', '--no_dist_averages',
        action= 'store_false',
        help= "whether to exclude distribution averages for global properties to scalar properties array.",
        required= False

    )
        # Title for plotting
    parser_b.add_argument(
        '-t', '--title',
        default= None,
        type= str,
        help= 'title to include in plots.',
        required= False
    )
        # Input networks
    requiredNamed = parser_b.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-i','--input',
        type= valid_path,
        help= "path to directory containing network files.",
        required= True,
    )

    ####################################################################################################################
    # create the parser for the "benchmark" command
    parser_c = subparsers.add_parser('benchmark', help='statistical benchmarking of inferences based on a Gold Standard.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_c.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 3
    )
    requiredNamed = parser_c.add_argument_group("required named arguments")
    requiredNamed.add_argument(
        '-gs','--gold_standard',
        type= valid_path,
        help= "path to gold standard network.",
        required= True,
    )
    requiredNamed.add_argument(
        '-inf','--inferences',
        type= valid_path,
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
        help= "Cutoff to use to compute the evaluation metrics. If False, the evaluation metrics are computed for every score in the inference.",
        required= False
    )
    parser_c.add_argument(
        '-dir', '--directed',
        action= 'store_true',
        help= 'whether gold standard and inferences are directed or not.',
        required= False
    )
    parser_c.add_argument(
        '-bl', '--baseline',
        action= 'store_true',
        help= 'whether to include the precision baseline or not.',
        required= False
    )
    parser_c.add_argument(
        '-k', '--keep_auc_coords_dicts',
        action= 'store_true',
        help= 'whether to return AUPR, AUROC values and precision, sensitivity and fpr datapoints for every inference in the benchmark.',
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
        help= "level of verbose to handle progress of process. Check logging levels for more information.",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
    parser_c.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files.",
        required= False
    )
    parser_c.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= "character used to separate columns in the network files.",
        required= False
    )
    parser_c.add_argument(
        '-o','--output',
        type= valid_path,
        default= os.getcwd(),
        help= "path to output directory, default for results printed to std.out.",
        required= False
    )

    ##############################################################################################################################
    # IMPORTANT Because a different formatter class is employed for subparser_d, DEFAULTS NEED TO BE ADDED MANUALLY TO HELP MESSAGES
    # create the parser for the "classify" command
    parser_d = subparsers.add_parser('classify', help='classification of networks into clusters and evaluation of said classification.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser_d.add_argument(
        '--runmode',
        type= int,
        help=argparse.SUPPRESS,
        default= 4
    )
        # Required named arguments (also mutually exclusive)
    # Usar textwrap para formatear la descripción del grupo
    group_description = ('''\
        ----------------------------------------------------------------------------------------------------------------
        This argument group includes the input for classify.
        There are three possible use cases:
                                        
            1. A distance dataframe has already been computed.
            2. Netective was previously run and a directory with the properties files exists.
            3. Properties computation is required.
        
        These are therefore mutually exclusive arguments.
        ''')
    requiredNamed = parser_d.add_argument_group('required named arguments', group_description)
    req_mut_exclusive = requiredNamed.add_mutually_exclusive_group(required= True)
        # Pre-computed distance dataframe
    req_mut_exclusive.add_argument(
        '-ddf', '--distance_dataframe',
        type= valid_path,
        help= 'path to file with distance dataframe pre-computed. IMPORTANT: if distance dataframe is provided, no results directory or networks directory can apply. Format has to be a dataframe where the first column serves as index and the first row serves as header. Delimiter and comments are specified with args.'
    )
        # Results of Netective pre-computed properties
    req_mut_exclusive.add_argument(
        '-rd', '--results_dir',
        type= valid_path,
        help= "directory path to files. ONLY if Netective's properties computation has already been performed. IMPORTANT: if results directory is provided, no distance dataframe or networks directory can apply."
    )
        # Networks directory files
    req_mut_exclusive.add_argument(
        '-nets', '--networks_dir',
        type= valid_path,
        help= 'directory path to files. ONLY if properties computation has not been performed yet. IMPORTANT: if networks directory is provided, no distance dataframe or results directory can apply.'
    )

        # Norm
    parser_d.add_argument(
        '-n','--normalization',
        default= None,
        help= "normalization method for structural properties. (default: None)",
        choices= ['network', 'biological'],
        required= False
    )
        # Direction
    parser_d.add_argument(
        '-dir', '--directed',
        action= 'store_true',
        help= 'whether the gold standard and inferences are directed or not. (default: False)',
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
            # 'Complex Feed-Forward Circuits',
            # 'Feed-Forward Circuits',
            'Max Degree',
            # 'Gini Index',
            'Global Efficiency',
            'Undirected Gini Index',
            'Entropy of Degree Distribution',
            # 'Self-Loops'
            ],
        required= False,
        help= "list of selected properties used for analysis, defaults to selected properties for best classification. Accepted format: coma-separated string, written between 's. (default: ['Average Local Efficiency', 'Radius', 'Center', 'Periphery', 'Complex Feed-Forward Circuits', 'Feed-Forward Circuits', 'Max Degree', 'Gini Index', 'Global Efficiency', 'Undirected Gini Index', 'Entropy of Degree Distribution', 'Self-Loops'])"
    )
        # Workers
    parser_d.add_argument(
        '-w','--workers',
        type= valid_workers,
        default= '2',
        help= "number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of usable threads. Only applies if directory has more than one readable network. (default: 2)",
        required= False
    )
        # Keep averages
    parser_d.add_argument(
        '-avg', '--add_averages',
        action= 'store_true',
        help= 'whether to include the averages of local properties in the global properties array. (default: False)',
        required= False
    )
        # Association metric
    parser_d.add_argument(
        '-ass', '--association_metric',
        type= str,
        choices= ['pearson', 'spearman', 'cosine'],
        help= "correlation metric for distance calculation. (default: 'pearson')",
        required= False,
        default= 'pearson'
    )
        # Not required mutually exclusive arguments
    group_description = ('''\
        ----------------------------------------------------------------------------------------------------------------
        This argument group includes arguments for forming flat clusters.
        There are several possible criterions, here we consider two:
                                        
            - max number of clusters: "Finds a minimum threshold r so that the cophenetic distance between any two original observations in the same flat cluster
                                        is no more than r and no more than t flat clusters are formed."
            - threshold distance: "Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t"
        
        These are therefore mutually exclusive arguments.
        See scipy.cluster.hierarchy.fcluster() for more info.
        ''')
    not_req_mut_group = parser_d.add_argument_group('not required mutually exclusive arguments', group_description)
    not_req_mut_exclusive = not_req_mut_group.add_mutually_exclusive_group(required=False)
        # Clusters
    not_req_mut_exclusive.add_argument(
        '-cl','--clusters',
        default= None,
        type= int,
        help= "max number of clusters to classify networks into. IMPORTANT: if number of clusters is given, then no threshold for distance can apply. (default: None)",
        required= False
    )
        # Threshold distance
    not_req_mut_exclusive.add_argument(
        '-t','--threshold',
        default= 0.7,
        type= restricted_float,
        help= "t distance for clustering, threshold to apply when forming flat clusters. IMPORTANT: if a threshold is given, no max number of clusters can apply. (default: 0.7)",
        required= False
    )
        # Distance metric
    parser_d.add_argument(
        '-mtr','--metric',
        default= 'euclidean',
        type= str,
        help= "distance metric to use. See scipy.spatial.distance.pdist() for more info. (default: 'euclidean')",
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
        # Clustering method
    parser_d.add_argument(
        '-m','--method',
        default= 'ward',
        type= str,
        help= "linkage method to use for calculating clusters. See scipy.clusters.hierarchy.linkage() for more info. (default: 'ward')",
        choices= ['single', 
                  'complete', 
                  'average', 
                  'weighted', 
                  'centroid', 
                  'median', 
                  'ward'],
        required= False
    )
        # Map IDs
    parser_d.add_argument(
        '-mids', '--map_ids',
        action= 'store_true',
        help= 'whether to return the clusters as dictionaries to map IDs of members. (default: False)',
        required= False
    )
        # Verbose
    parser_d.add_argument(
        '-v','--verbose',
        type= str,
        default= 'WARNING',
        help= "level of verbose to handle progress of process. Check logging levels for more information. (default: 'WARNING')",
        choices= ['DEBUG', 'INFO','WARNING', 'ERROR', 'CRITICAL'],
        required= False
    )
        # Nets files format
    parser_d.add_argument(
        '-nff', '--net_f_format',
        type= str,
        help= "nets files format to parse. (default: 'edgelist')",
        choices= ['edgelist', 'graphml', 'adj list', 'multiline adj list'],
        required= False,
        default= 'edgelist'
    )
        # Comments character in networks files
    parser_d.add_argument(
        '-c','--comments',
        type= str,
        default= "#",
        help= "character used to indicate comments in the network files. (default: '#')",
        required= False
    )
        # Delimiter to parse networks
    parser_d.add_argument(
        '-d','--delimiter',
        type= str,
        default= '\t',
        help= r"character used to separate columns in the network files. Also referred to character used as delimiter if a distance dataframe is provided. (default: '\t')",
        required= False
    )
        # Path to dump outputs
    parser_d.add_argument(
        '-o','--output',
        type= valid_path,
        default= os.getcwd(),
        help= f"path to output directory. (default: {os.getcwd()})",
        required= False
    )
    
    args = parser.parse_args()
    args.delimiter = args.delimiter.encode("utf-8").decode("unicode_escape")

    return args
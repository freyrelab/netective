Netective is a computational tool that enables similarity analysis from a comprehensive structural characterization. Netective also allows an optimized statistical benchmarking between network inferences and a gold standard.  
Netective can be used from the command line and through Python. For Python's tutorial, see ...

To use netective command line, first thing to do is activate conda env

```
conda activate netective
```

For first-time users, it is recommended to read the help options using `netective --help` This will show the following message: 

```
usage: netective [-h] {characterize,compare,benchmark,classify} ...

        ( ◣_◢ )
        Welcome to Netective!
        Computational tool for network similarity analysis. Netective is capable of performing
        a comprehensive structural characterization of unweighted graphs from all domains, directed and/or undirected.
        Netective can then compare an unlimited number of networks through said structural characterization, as well as
        classifying them into clusters using two possible criterions.
        Netective can also perform an optimized statistical benchmarking of network inferences based on a Gold Standard.

        Netective was developed and is maintained by the team at FreyreLab, Center for Genomic Sciences, National Autonomous University of Mexico (UNAM).
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


optional arguments:
  -h, --help            show this help message and exit

Runmodes:
          Netective has four main runmodes, each with different and specific arguments.
          Please consult each runmode's help manual before running any of them.
          Enjoy Netective!


  {characterize,compare,benchmark,classify}
    characterize        characterization of inputed network/s through structural properties.
    compare             comparison between multiple networks based on their topologies.
    benchmark           statistical benchmarking of inferences based on a Gold Standard.
    classify            classification of networks into clusters and evaluation of said classification.
```

As said by the help mesage netective in comand line has 4 diferent modules, each of them has a different usage.

## Note on verbose

Each of the run-modes has a verbose parameter with five different levels, which can be set via using `-v` parameter of the corresponding function to monitor the progress of each process with varying levels of detail:

- `Debug`: Provides the most detailed output, including internal steps and intermediate values—useful for debugging or in-depth inspection.
- `Info`: Simple but useful explanation of the progress of the process.
- `Warning`: Shows non-critical issues that may affect results or performance but do not stop execution.
- `Error`: Reports errors that prevent specific parts of the function from running correctly.
- `Critical`: Displays messages indicating serious failures.

If the user does not provide any of the above, the verbose level is set to Warning.

## Note on output files

All text output files begin with a commented header line. The comment character used is the one specified by the comment parameter. This header contains the complete Netective command used to generate the resulting file, including all input parameters and their values, as well as any default parameters that were applied.

## Characterize

The first run mode is `characterize` which realizes the structural characterization of all input networks.

#### Help message and usage details (inputs & outputs)

For first-time users, it is recommended to read the help options using `netective characterize --help` that'll show the following message: 

```
usage: netective characterize [-h] [-n {network,biological}] [-dir] [-p SELECTED_PROPS] [-w WORKERS] [-k] [-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                              [-nff {edgelist,graphml,adj list,multiline adj list}] [-c COMMENTS] [-d DELIMITER] [-o OUTPUT] -i INPUT

optional arguments:
  -h, --help            show this help message and exit
  -n {network,biological}, --normalization {network,biological}
                        normalization method for structural properties. (default: None)
  -dir, --directed      whether input networks are directed or not. (default: False)
  -p SELECTED_PROPS, --selected_props SELECTED_PROPS
                        list of selected properties used for analysis. Accepted format: coma-separated string, written between "s. (default: ['all'])
  -w WORKERS, --workers WORKERS
                        number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also
                        the max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical
                        detection of usable threads. Only applies if directory has more than one readable network. (default: 2)
  -k, --keep_props      whether to save dataframes of the properties values for each network analyzed. (default: False)
  -v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        level of verbose to handle progress of process. Check logging levels for more information. (default: WARNING)
  -nff {edgelist,graphml,adj list,multiline adj list}, --net_f_format {edgelist,graphml,adj list,multiline adj list}
                        nets files format to parse. (default: edgelist)
  -c COMMENTS, --comments COMMENTS
                        character used to indicate comments in the network file(s). (default: #)
  -d DELIMITER, --delimiter DELIMITER
                        character used to separate columns in the network file(s). (default: )
  -o OUTPUT, --output OUTPUT
                        path to output directory. (default: ./)

required named arguments:
  -i INPUT, --input INPUT
                        path to folder containing network file(s). (default: None)
```

The program only accepts folders as input. If you want to characterize only one network, you must create a folder containing that network.  
For each network file within the specified folder, two different output files will be generated. The first, `filename_scalars_props.png`, will be a histogram of the global properties, and the second, `filename_distributions_props.png`, is an image showing the different distributions obtained from the local properties. If you decide to keep the properties, the same-named file with a `.txt` extension will be generated containing selected properties distributions.

### Basic usage

For the most basic usage of netective characterize module, the only required argument is the input directory, assuming that your input networks-file have deffault parameters, such as a tab-separated edgelist format with comments marked as `#`.

```
netective characterize -i ./input_directory/
```

To customize any of the above-mentioned parameters, use (where `-d` represents the delimiter of the file, `-c` is the first character in a comment line, and `-nff` is the file format):

```
netective characterize -i ./input_directory/ -d '\t' -c # -nff edgelist
```

You may also change the output dir were images will be sabed from the current directoy `./` to any directory by adding `-o` as another parameter.

```
netective characterize -i ./input_directory/ -d '\t' -c # -nff edgelist -o ./output_directory/
```

If you want to keep properties' values instead of plotting them, add the parameter `-k`.

```
netective characterize -i ./input_directory/ -d '\t' -c # -nff edgelist -o ./output_directory/
```

### Advanced usage

Netective allows parallelization of processes so a user may introduce the number of workers to use for the analysis with `-w` as a parameter.

```
netective characterize -i ./input_directory/ -w 2
```

Default analysis includes all available network properties to compute. 

List of node-level Properties:
- Average Degree for Nearest Neighbors
- Average Out-Degree for Nearest Neighbors
- Clustering Coefficient
- Degree
- In-Degree
- Out-Degree
- Locality Index
- Rich Club Coefficient
- Subgraph Centrality
- Betweenness Centrality
- Eccentricity

List of Global properties:
- Density
- Undirected Density
- Average Clustering Coefficient
- Feed-Forward Circuits
- Complex Feed-Forward Circuits
- 3-Feedback Loops
- Entropy of Degree Distribution
- Entropy of Out-Degree Distribution
- Max Degree
- Max In-Degree
- Max Out-Degree
- Giant component size
- Gini Index
- Undirected Gini Index
- Non-leaf nodes
- Self-loops
- Undirected Self-Loops
- Global Efficiency
- Average Local Efficiency
- Average Shortest Path Length
- Diameter  
- Center
- Periphery  
- Radius
- Number of Edges
- Number of Arcs
- Number of Nodes

The user may decide which of these properties to calculate using the `-p` parameter followed by a comma-separated list. 

```
netective characterize -i ./input_directory/ -w 2 -p "Giant Component Size,Average Local Efficiency,Average Shortest Path Length,Clustering Coefficient"
```

To conserve props values and not plot the results, you can also add the argument `-k`.
## Compare

The second run mode is `compare`, which realizes the structural characterization of all input networks to compare their properties vectors and compute the distances between them. Users may generate random analog networks preserving selected properties to facilitate comparative analyses.
#### Help message and usage details (inputs & outputs)

For first-time users, it is recommended to read the help options using `netective compare --help` that'll show the following message: 

```
usage: netective compare [-h] [-n {network,biological}] [-dir] [-p SELECTED_PROPS] [-w WORKERS] [-k] [-a {pearson,spearman,cosine}]
                         [-mtr https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist]
                         [-m https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html] [-im INCMODELS] [-c2m] [-nm N_MODELS] [-dirm]
                         [-ba M4BA] [-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-nff {edgelist,graphml,adj list,multiline adj list}] [-c COMMENTS] [-d DELIMITER]
                         [-o OUTPUT] [-noa] [-t TITLE] -i INPUT

optional arguments:
  -h, --help            show this help message and exit
  -n {network,biological}, --normalization {network,biological}
                        normalization method for structural properties. (default: None)
  -dir, --directed      whether input networks are directed or not. (default: False)
  -p SELECTED_PROPS, --selected_props SELECTED_PROPS
                        list of selected properties used for analysis. Accepted format: coma-separated string, written between "s. (default: ['all'])
  -w WORKERS, --workers WORKERS
                        number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also the
                        max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of
                        usable threads. Only applies if directory has more than one readable network. (default: 2)
  -k, --keep_props      whether to save dataframes of the properties values for each network analyzed. (default: False)
  -a {pearson,spearman,cosine}, --association {pearson,spearman,cosine}
                        association metric for calculating correlation between scalar properties arrays. (default: pearson)
  -mtr https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist, --metric https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist
                        distance metric to use. See scipy.spatial.distance.pdist() for more info. (default: euclidean)
  -m https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html, --method https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
                        linkage method to use for calculating clusters. See scipy.clusters.hierarchy.linkage() for more info. (default: ward)
  -im INCMODELS, --incmodels INCMODELS
                        which random network generators to use to create analogs to each input network for comparison to them. Accepted format: coma-separated
                        string, written between "s. Admitted models are: Erdos GNP, Erdos GNM, K Regular and Barabasi Albert. (default: None)
  -c2m, --comp2models   whether to create a comparison heatmap of input networks to model analogs. Default behavior is to create symmetric heatmap of all analyzed
                        networks (input and analogs). (default: False)
  -nm N_MODELS, --n_models N_MODELS
                        number of analog networks from each random model to create for each input network. (default: 2)
  -dirm, --directed_models
                        whether analog random models will be created using direction, if possible. (default: False)
  -ba M4BA, --m4ba M4BA
                        m to use in Barbasi Albert algorithm. It can either be positive integers or degree distributions from each input network. Accepted format:
                        coma-separated string, written between "s. Admitted distributions: in degree, out degree and undirected degree. String may include several
                        positive integers aswell as any or all available distributions. (default: [2])
  -v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        level of verbose to handle progress of process. Check logging levels for more information. (default: WARNING)
  -nff {edgelist,graphml,adj list,multiline adj list}, --net_f_format {edgelist,graphml,adj list,multiline adj list}
                        nets files format to parse. (default: edgelist)
  -c COMMENTS, --comments COMMENTS
                        character used to indicate comments in the network files. (default: #)
  -d DELIMITER, --delimiter DELIMITER
                        character used to separate columns in the network files. (default: )
  -o OUTPUT, --output OUTPUT
                        path to output directory. (default: ./)
  -noa, --no_dist_averages
                        whether to exclude distribution averages for global properties to scalar properties array. (default: True)
  -t TITLE, --title TITLE
                        title to include in plots. (default: None)

required named arguments:
  -i INPUT, --input INPUT
                        path to directory containing network files. (default: None)

```

This netective run mode works just as `characterize`, only accepting folders as input. The structural characterization of a single network through this run mode is possible but not recommended. Default runs require more than one network and a couple of properties to ensure a comparison can be computed.

Normal output for this function includes
- `nets_comparison.png`: Comparative clustermap of the distances between the properties vectors of each input network between them or to the mean properties of random theoretical analogs from the available model options (`Erdos GNP`, `Erdos GNM`, `K Regular`, and `Barabasi Albert`).
- `association_df.ext`: dataframe with all distances between the networks (same as the one plotted in the image `nets_comparison.png`). File extensions can change depending on the selected delimiter of the input network (TSV for tab-separated files, CSV for comma-separated files, or TXT for other delimiters).

Saving properties files:  
If you'd rather have property values instead of plotting them by using the `-k` parameter, you'll get both a `net_filename_distributions_props.txt` and a `net_filename_scalars_props.txt` for each network. When random graph analogs are generated, they'll also be saved into two different files, each containing the mean value of all scalars and the mean distribution moments' values. These file names would start with the model used for their creation.

### Basic & Advanced usage

Netectives compare basic usage, which is the same as characterize (see section above). The main difference between them is the output results, explained in the `Help message and usage details (inputs & outputs)` of both sections in the tutorial.

### Network's properties vector

Each network's properties vector is created using all their scalar properties and the average of the node-level properties, the last one being an optional part, which can be deprecated by using the `-noa argument`.

```
netective characterize -i ./input_directory/ -noa
```

### Size-independent comparison

Netectives compare module includes an option to compute a normalization of the properties to allow a size-independent analysis. You can choose whether you'd like a theoretical normalization (following classical graph theory) or a biological normalization (which focuses on the structure of regulatory networks).

```
## --> Clasical normalization
netective characterize -i ./input_directory/ -norm network

## --> Biological normalization
netective characterize -i ./input_directory/ -norm biological
```

### Calculating distances and clustering

To calculate the distances between each property vector, the user can choose between one of three metrics: `pearson`, `spearman`, or `cosine`, with `pearson` being the default option.

```
netective characterize -i ./input_directory/ -a pearson
```

The results image is a clustermap of the distances between networks. Parameters of the clustering, such as [method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) and [metric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist), can be modified by the `-m` and `-mtr` parameters, respectively. Default clustering parameters are the `euclidean` metric and `ward` method.

```
netective characterize -i ./input_directory/ -m ward -mtr euclidean
```

### Random networks usage 

Netective allows users to generate random analogs based on the original networks' properties such as number of edges, number of nodes, density, or their connectivity distribution. There are 4 models available that can be chosen using the `-im` argument. For default only 2 random analogs are generated, and the mean property values between them plus their distribution average are used to create the property vector for comparison. If you want to generate more networks, modify the `-nm` argument. 

```
netective characterize -i ./input_directory/ -im "Erdos GNP,Erdos GNM,K Regular,Barabasi Albert" -nm 2
```

Models can be created with and without directions depending on whether the `-dirm` argument is added.

```
netective characterize -i ./input_directory/ -im "Erdos GNP,Erdos GNM,K Regular, Barabasi Albert" -nm 2 -dirm
```

Barabási-Albert algorithm can generate different graphs depending on the `m` parameter utilized; the default value is set to `2`. To modify this parameter, use the `-ba` argument; you can include all network degree distributions (`in-degree`, `out-degree`, and `undirected degree`) plus any integer you'd like.

```
netective characterize -i ./input_directory/ -im "Erdos GNP, Erdos GNM, K Regular, Barabasi Albert" -nm 2 -dirm -ba "in degree,out degree,undirected degree,2,3,5"
```

The resulting distance matrix from the command above is a square matrix with a size of `(n+m) x (n+m)`, where `n` represents the total number of input networks and `m` represents the number of models generated, including all Barabasi variations. This run mode also enables the possibility to compute the comparison from each input network to their corresponding analogs by adding the`-c2m` parameter (ending in an `n x m` distance matrix).

```
netective characterize -i ./input_directory/ -im "Erdos GNP, Erdos GNM, K Regular, Barabasi Albert" -nm 2 -dirm -ba "in degree,out degree,undirected degree,2,3,5" -c2m
```

## Classify

The third run mode is `classify`, 
#### Help message and usage details (inputs & outputs)

For first-time users, it is recommended to read the help options using `netective classify --help` that'll show the following message: 

```

usage: netective classify [-h] (-ddf DISTANCE_DATAFRAME | -rd RESULTS_DIR | -nets NETWORKS_DIR) [-n {network,biological}] [-dir] [-p SELECTED_PROPS] [-w WORKERS]
                          [-avg] [-ass {pearson,spearman,cosine}] [-cl CLUSTERS | -t THRESHOLD]
                          [-mtr {braycurtis,canberra,chebyshev,cityblock,correlation,cosine,dice,euclidean,hamming,jaccard,jensenshannon,kulczynski1,mahalanobis,matching,minkowski,rogerstanimoto,russellrao,seuclidean,sokalmichener,sokalsneath,sqeuclidean,yule}]
                          [-m {single,complete,average,weighted,centroid,median,ward}] [-mids] [-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                          [-nff {edgelist,graphml,adj list,multiline adj list}] [-c COMMENTS] [-d DELIMITER] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -n {network,biological}, --normalization {network,biological}
                        normalization method for structural properties. (default: None)
  -dir, --directed      whether the gold standard and inferences are directed or not. (default: False)
  -p SELECTED_PROPS, --selected_props SELECTED_PROPS
                        list of selected properties used for analysis, defaults to selected properties for best classification. Accepted format: coma-separated
                        string, written between 's. (default: ['Average Local Efficiency', 'Radius', 'Center', 'Periphery', 'Complex Feed-Forward Circuits', 'Feed-
                        Forward Circuits', 'Max Degree', 'Gini Index', 'Global Efficiency', 'Undirected Gini Index', 'Entropy of Degree Distribution', 'Self-Loops'])
  -w WORKERS, --workers WORKERS
                        number of workers to use for parallelization of properties characterization, default is minimal parallelization. IMPORTANT: it is also the
                        max number of networks loaded simultaneously into memory at the same time at any given moment. IMPORTANT: auto for automatical detection of
                        usable threads. Only applies if directory has more than one readable network. (default: 2)
  -avg, --add_averages  whether to include the averages of local properties in the global properties array. (default: False)
  -ass {pearson,spearman,cosine}, --association_metric {pearson,spearman,cosine}
                        correlation metric for distance calculation. (default: 'pearson')
  -mtr {braycurtis,canberra,chebyshev,cityblock,correlation,cosine,dice,euclidean,hamming,jaccard,jensenshannon,kulczynski1,mahalanobis,matching,minkowski,rogerstanimoto,russellrao,seuclidean,sokalmichener,sokalsneath,sqeuclidean,yule}, --metric {braycurtis,canberra,chebyshev,cityblock,correlation,cosine,dice,euclidean,hamming,jaccard,jensenshannon,kulczynski1,mahalanobis,matching,minkowski,rogerstanimoto,russellrao,seuclidean,sokalmichener,sokalsneath,sqeuclidean,yule}
                        distance metric to use. See scipy.spatial.distance.pdist() for more info. (default: 'euclidean')
  -m {single,complete,average,weighted,centroid,median,ward}, --method {single,complete,average,weighted,centroid,median,ward}
                        linkage method to use for calculating clusters. See scipy.clusters.hierarchy.linkage() for more info. (default: 'ward')
  -mids, --map_ids      whether to return the clusters as dictionaries to map IDs of members. (default: False)
  -v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        level of verbose to handle progress of process. Check logging levels for more information. (default: 'WARNING')
  -nff {edgelist,graphml,adj list,multiline adj list}, --net_f_format {edgelist,graphml,adj list,multiline adj list}
                        nets files format to parse. (default: 'edgelist')
  -c COMMENTS, --comments COMMENTS
                        character used to indicate comments in the network files. (default: '#')
  -d DELIMITER, --delimiter DELIMITER
                        character used to separate columns in the network files. Also referred to character used as delimiter if a distance dataframe is provided.
                        (default: '\t')
  -o OUTPUT, --output OUTPUT
                        path to output directory. (default: ./)

required named arguments:
          ----------------------------------------------------------------------------------------------------------------
          This argument group includes the input for classify.
          There are three possible use cases:

              1. A distance dataframe has already been computed.
              2. Netective was previously run and a directory with the properties files exists.
              3. Properties computation is required.

          These are therefore mutually exclusive arguments.


  -ddf DISTANCE_DATAFRAME, --distance_dataframe DISTANCE_DATAFRAME
                        path to file with distance dataframe pre-computed. IMPORTANT: if distance dataframe is provided, no results directory or networks directory
                        can apply. Format has to be a dataframe where the first column serves as index and the first row serves as header. Delimiter and comments are
                        specified with args.
  -rd RESULTS_DIR, --results_dir RESULTS_DIR
                        directory path to files. ONLY if Netective's properties computation has already been performed. IMPORTANT: if results directory is provided,
                        no distance dataframe or networks directory can apply.
  -nets NETWORKS_DIR, --networks_dir NETWORKS_DIR
                        directory path to files. ONLY if properties computation has not been performed yet. IMPORTANT: if networks directory is provided, no distance
                        dataframe or results directory can apply.

not required mutually exclusive arguments:
          ----------------------------------------------------------------------------------------------------------------
          This argument group includes arguments for forming flat clusters.
          There are several possible criterions, here we consider two:

              - max number of clusters: "Finds a minimum threshold r so that the cophenetic distance between any two original observations in the same flat cluster
                                          is no more than r and no more than t flat clusters are formed."
              - threshold distance: "Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t"

          These are therefore mutually exclusive arguments.
          See scipy.cluster.hierarchy.fcluster() for more info.


  -cl CLUSTERS, --clusters CLUSTERS
                        max number of clusters to classify networks into. IMPORTANT: if number of clusters is given, then no threshold for distance can apply.
                        (default: None)
  -t THRESHOLD, --threshold THRESHOLD
                        t distance for clustering, threshold to apply when forming flat clusters. IMPORTANT: if a threshold is given, no max number of clusters can
                        apply. (default: 0.7)

```

This run mode allows the user to obtain a clustering of a group of networks based on distances computed from the properties vector. Netective `classify` receives one of three inputs:
- A path to a file containing a distance matrix
- The results directory for the properties computed by compare module
- A network directory. If a network directory is provided, the user must add all the arguments to ensure the characterization is produced as intended. Please see the following `compare tutorial` sections for more details: `Basic & Advanced usage`, `Network's properties vector`, `Size-independent comparison`.

whenether input it's introduced the output remains the same, a two column file were every row is a network and its corresponding cluster after the clasification. 

### Basic usage (data input)

The input distance matrix can be generated by Netective or obtained from an external source. To use a distance matrix as input, specify the `-dff` argument. As with network input files, the distance matrix dataframe may use different delimiters, which can be specified using the `-d` argument.

```
netective classify -dff ./input_distance_df.tsv -d '\t'
```

To use the results directory from a previous Netective analysis, use the `-rd` argument.

```
netective classify -rd ./previous_netective_output_dir/
```

To save results in an specific directory use `-o`

```
netective classify -rd ./previous_netective_output_dir/ -o ./results_dir/
```

### Using networks dir as input

To use a directory containing network files, specify the `-nets` argument.

```
netective classify -nets ./network_dir/
```

For detailed usage examples, refer to the following sections of the `Compare` tutorial: `Basic & Advanced Usage`, `Network Properties Vector`, and `Size-Independent Comparison`.

Note: The `-noa` argument, which removes average values from local property distributions when generating property vectors, has been replaced in this module by the `-avg` argument. Unlike `-noa` in the Compare run mode, whose presence excluded these properties, the `-avg` argument in this run mode must be explicitly provided to include them.

If the user chooses to use the classify run mode with a network directory as input, the distances between property vectors must be computed. Three distance metrics are available for this purpose: `pearson`, `spearman`, and `cosine`, with `pearson` being the default option.

```
netective characterize -nets ./network_dir/ -ass pearson
```

### Network clasification through clusters

To create the clusters both [method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) and [metric](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist), can be modified by the `-m` and `-mtr` parameters, respectively. Default clustering parameters are the `euclidean` metric and `ward` method.

```
netective classify -rd ./previous_netective_output_dir/ -m ward -mtr euclidean
```

To separate networks into groups, the user can choose between two approaches:

1. Cut the cluster dendrogram at a specified cophenetic distance threshold.
2. Specify the desired number of groups.

By default, networks are grouped using a cophenetic distance threshold of `0.7`.

```
# Using a distance threshold
netective classify -rd ./previous_netective_output_dir/ -m ward -mtr euclidean -t 0.7

# Using a fixed number of groups
netective classify -rd ./previous_netective_output_dir/ -m ward -mtr euclidean -cl 3
```

If both `-t` and `-cl` are provided, the classification will be performed using the number of groups specified by `-cl`.

## Benchmark

The last run mode is `benchmark`, which computes the statistical evaluation of predicted/inferred networks, where each edge carries a weight representing the confidence of the given interaction. You can evaluate either a single network or a set of networks **against the same gold standard**.
#### Help message and usage details (inputs & outputs)

For first-time users, it is recommended to read the help options using `netective benchmark --help` that'll show the following message: 

```
usage: netective benchmark [-h] -gs GOLD_STANDARD -inf INFERENCES [-dir] [-g] [-sl] [-coff CUTOFF] [-ocoff] [-f1] [-mcc] [-bl] [-k] [-s]
                           [-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-c COMMENTS] [-d DELIMITER] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -dir, --directed      whether gold standard and inferences are directed or not. (default: False)
  -g, --greater_is_better
                        whether the inference score is better when it is higher or lower. If greater_is_better is not declared, the lower the score the better the
                        inference. (default: False)
  -sl, --self_loops     whether the self-loops are allowed or not. (default: False)
  -coff CUTOFF, --cutoff CUTOFF
                        cutoff to use to compute the evaluation metrics. If False, the evaluation metrics are computed for every score in the inference. (default:
                        False)
  -ocoff, --optimal_cutoff
                        whether to plot the optimal cutoff for each inference. (default: False)
  -f1, --f1_score       whether to plot the F1 score for each inference. (default: False)
  -mcc, --matthews_corr_coeff
                        whether to plot the Matthews Correlation Coefficient for each inference. (default: False)
  -bl, --baseline       whether to include the precision baseline or not. (default: False)
  -k, --keep_auc_coords_dicts
                        whether to return metrics and distributions calculated for every inference in the benchmark. IMPORTANT: no plotting will be perfomed if this
                        option is selected. (default: False)
  -s, --score           whether the inference networks have a score attribute or not. (default: False)
  -v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        level of verbose to handle progress of process. Check logging levels for more information. (default: WARNING)
  -c COMMENTS, --comments COMMENTS
                        character used to indicate comments in the network files. (default: #)
  -d DELIMITER, --delimiter DELIMITER
                        character used to separate columns in the network files. (default: )
  -o OUTPUT, --output OUTPUT
                        path to output directory, default for results printed to std.out. (default: ./)

required named arguments:
  -gs GOLD_STANDARD, --gold_standard GOLD_STANDARD
                        path to gold standard network. (default: None)
  -inf INFERENCES, --inferences INFERENCES
                        path to directory containing inferenced networks. (default: None)

```

The only required arguments for this run mode are the path to the gold-standard network file and a path to the directory of the inference network (whether there is one or more inferences).

The standard output are 4 image files:
- `aupr.png`: This image contains a bar plot where the x-axis represents the complete inference AUPR value and each bar represents an inference network and the baseline value if included.
- `pr_curves.png`: This image contains all precision-recall curves computed. The best inference is colored in red and labeled on the upper right side of the plot.
- `auroc.png`: This image contains a bar plot where the x-axis represents the complete inference AUROC values and each bar represents an inference network and the baseline value if included.
- `roc_curves.png`: This image contains all ROC curves computed. The best inference is colored in red and labeled on the upper right side of the plot.  

If other parameters are selected, other image files may be included, such as:
- `optimal_cutoffs.png`: This image contains a bar plot where the x-axis represents the size of the inference that maximizes the f1 score. Each bar represents an inference network and the baseline value if included.
- `f1_scores.png`: This image contains a bar plot where the x-axis represents the complete inference f1-score value and each bar represents an inference network and the baseline value if included.
- `mcc.png`: This image contains a bar plot where the x-axis represents the complete Matthews correlation coefficient value and each bar represents an inference network and the baseline value if included.

If the `-k` parameter is chosen, the file output changes to the following:
- `aupr_scores.tsv`: This file contains a 2-column dataframe, where the first column is the inference name and the second is its corresponding AUPR score.
- `auroc_scores.tsv`: This file contains a 2-column dataframe, where the first column is the inference name and the second is its corresponding AUROC score.
- `fpr.tsv`: This file contains 2 lines per inference, the first one is a header identified by the characters `>>>` followed by the inference name and the text `false positive rate datapoints`. The next line contains the complete distribution of the false positive rate.
- `precision.tsv`: This file contains 2 lines per inference, the first one is a header identified by the characters `>>>` followed by the inference name and the text `precision datapoints`. The next line contains the complete distribution of the precision scores.
- `scores_metrics_distributions.tsv`: This file contains eight lines per inference. The first line is a header identified by the characters `>>>` the inference name, and the text `inference scores`. The second line includes the inference edge's score evaluated at each point. The third line has the header `>>>Inference_name F1 scores`, followed in the next line by the F1 values' scores distribution. The fifth and seventh lines contain the headers `>>>Inference_name Matthews Correlation Coefficient values` & `>>>Inference_name Accuracy` scores followed by their corresponding distributions.
- `sensitivity.tsv`: This file contains 2 lines per inference, the first one is a header identified by the characters `>>>` followed by the inference name and the text `sensitivity datapoints`. The next line contains the complete distribution of the sencitivity scores.
- `stats_summary.tsv`: This file contains a dataframe where rows represent every inference and columns the score of each metric computed (AUPR, AUROC, F1 score, MCC, optimal cutoff, accuracy).  

All file values/distributions are tab-delimited (in case no other delimiter is chosen with parameter `-d`).

### Usage

Most basic usage is:

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/
```

Result files are saved in the same directory where the command is run. To change the output directory, use the `-o` argument.

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -o ./out_dir/
```

Netective benchmark module assumes network files are tab-separated, with comments beginning with `#`. If input graphs do not follow that pathern, delimither and coment characters can be modified with `-d` and `-c` parameters respectively. Whether files are directed or not should be specified by adding the -dir option (if not introduced, networks are computed without direction).

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -d '\t' -c'#' -dir
```

Inference networks' edge scores are taken by their position in the file;  lower positions get lower scores. When inference networks have precomputed scores, it has to be specified with the `-s` parameter. By default scores are considered the lower the better, regardless of whether they are precalculated or not. To specify the opposite, add the `-g` parameter.

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -d '\t' -c'#' -dir -s -g
```

A baseline result can also be computed whether there's one or more inference networks by adding the `-bl` argument.

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -d '\t' -c'#' -dir -s -g -bl
```

Netective benchmark run-mode default output is images with the scores of AUPR and AUROC and their corresponding distributions (see this tutorial benchmark's `Help message and usage details (inputs & outputs)` section for more details), but the tool also allows the user to know the values of other metrics such as F1-score, Matthews Correlation Coefficient, and the optimal cutoff of each inference. To also compute these metrics, add the parameters `-f1`, `-mcc`, and `-ocoff`, respectively.

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -d '\t' -c'#' -dir -s -g -bl -ocoff -f1 -mcc
```

If you prefer conserving the distributions rather than generating plots and images, you can add the `-k` option (to see all file outputs, see this tutorial benchmark's `Help message and usage details (inputs & outputs)` ).

```
netective benchmark -gs ./gold_standard.txt -inf ./inferences_dir/ -d '\t' -c'#' -dir -s -g -bl -k
```
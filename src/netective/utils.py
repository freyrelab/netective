from __future__ import annotations

import matplotlib.figure

"""Utility functions for the netective package."""

# __all__ = [
#     "concat_path",
#     "run_parallel",
#     "validate_network",
#     "parse_nets",
#     "flatten_list_of_iterables",
#     "compute_moments",
# ]

import os
import re
import sys
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import math as m
import networkx as nx
import igraph as ig
from tqdm import tqdm
import concurrent.futures
from itertools import chain
from collections import defaultdict
# TODO Incluir las demas correlaciones admitidas (tiene que regresar tupla)
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Union, Callable, Iterable

from netective.logging_info import get_logger, set_log_level
from netective.structure.dataviz import create_comp_heatmap

import matplotlib

concat_path = os.path.join

utils_logger = get_logger(__name__)

def cosine_similarity(x, y):
    if len(x) != len(y):
        raise ValueError("both arrays must have the same length")
    return m.fsum(x * y) / m.sqrt(m.fsum(x ** 2) * m.fsum(y ** 2))

CORRELATIONS = {
    'pearson' : pearsonr,
    'spearman' : spearmanr,
    'cosine' : cosine_similarity
}

class NullGraphError(Exception):
    """Exception raised for null graph."""
    pass


def run_parallel(f, my_iter, workers, verbose: str = None):

    """
    Start the parallel processes.

    Parameters
    ----------
    f (function): Function to be executed in parallel.
    my_iter (Iterable): Iterable with the inputs for f.
        Each element of iterable will be unzipped before calling f.
    workers (int): Numer of processes to run in parallel.

    Returns
    -------
    Results: zip object.
        Contains the results of the function f.
    """
    warnings.filterwarnings('once')
    if verbose != None:
        current_level = utils_logger.getEffectiveLevel()
        set_log_level(utils_logger, verbose)
    
    my_iter = list(zip(*my_iter))
    len_iter = len(my_iter)
    with tqdm(total=len_iter, file=sys.stdout) as pbar:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for arg in my_iter:
                    name = arg[1]
                    utils_logger.info(f"Running {name}...")
                    futures[executor.submit(f, *arg)] = name

                results = defaultdict(dict)
                for future in concurrent.futures.as_completed(futures):
                    # try:
                    scalar, dist = future.result()
                    results["scalars"].update(scalar)
                    results["distributions"].update(dist)
                    utils_logger.warning(f'Finilized: {futures[future]}')
                    pbar.update(1)

                    # except Exception as exc:
                    #     print(f"Error: {exc}")
        except NotImplementedError as e:
            utils_logger.critical(e.message)
    if verbose != None:
        set_log_level(utils_logger, current_level)

    warnings.resetwarnings()
    return results

def get_allocated_memory(snapshot, key_type='lineno', filtered: bool = True):
    if filtered:
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(True, '*networkx*')
        ))
    top_stats = snapshot.statistics(key_type)
    return ((sum(stat.size for stat in top_stats)) / 1024) * 0.001024

def sort_files(path: str):
    files_paths = []
    for root, dir, files in os.walk(path):
        if len(files) != 0:
            for f in files:
                files_paths.append(os.path.join(root,f))
    files_paths = sorted (
        files_paths,
        key = lambda x: os.stat(x).st_size,
        reverse=True
    )
    """
    for name_of_file in files_paths: 
        size_of_file  = os.stat(name_of_file).st_size  
        print(size_of_file, ' -->', name_of_file)
    """
    
    return files_paths

def validate_network(G: nx.DiGraph | nx.Graph) -> Union(nx.DiGraph, nx.Graph):
    """Validates the network and returns a DiGraph or Graph object."""
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        utils_logger.critical("G must be a DiGraph or a Graph")
        raise TypeError("G must be a DiGraph or a Graph")
    if G.size() == 0:
        utils_logger.critical(f"G must have at least one edge. It has {G.size()} edges.")
        raise TypeError(f"G must have at least one edge. It has {G.size()} edges.")
    return G

def parse_network(
    file_path, comments="#", delimiter="\t", directed=True, score=False, use_position_as_score=False, net_file_format= 'edgelist'
) -> Union(nx.DiGraph, nx.Graph):
    """
    Parse a network file and return a networkx.DiGraph or networkx.Graph depending on the directed parameter.

    Args:
        file_path (str): Path to the network file.
        comments (str): Comment character.
        delimiter (str): Delimiter character.
        directed (bool): If True, the network will be a DiGraph, otherwise it will be a Graph.
        score (bool): If True, the network will use the third column of the file as the score of the edge.
        use_position_as_score (bool): If True, the position of the edge in the file will be used as the score of the edge.
        net_file_format (str): format to read network from file. Accepted formats are: edgelist, graphml, adj list and multiline adj list.
    """
    if score and use_position_as_score:
        utils_logger.critical("score and use_position_as_score cannot be True at the same time.")
        raise ValueError("score and use_position_as_score cannot be True at the same time.")

    if net_file_format == 'edgelist':
        if not use_position_as_score:
            if score:
                # check if first line has 3 columns
                with open(file_path, "r") as f:
                    first_line = f.readline()
                    cols = first_line.strip().split(delimiter)
                    if len(cols) < 3:
                        utils_logger.critical(
                            f"File {file_path} does not have a score column. Set score=False."
                        )
                        raise ValueError(
                            f"File {file_path} does not have a score column. Set score=False."
                        )

            G = nx.read_edgelist(
                file_path,
                comments=comments,
                delimiter=delimiter,
                create_using=nx.DiGraph if directed else nx.Graph,
                data=(("score", float),) if score else False,
            )
        else:
            G = nx.DiGraph() if directed else nx.Graph()
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if line.startswith(comments):
                        continue
                    cols = line.strip().split(delimiter)
                    source = cols[0]
                    target = cols[1]
                    G.add_edge(source, target, score=i)
    
    elif net_file_format == 'graphml':
        G = nx.read_graphml(file_path)
    
    else:
        adj_readers = {
            'multiline adj list' : nx.read_multiline_adjlist,
            'adj list' : nx.read_adjlist
        }
        G = adj_readers[net_file_format](file_path, comments= comments, delimiter= delimiter, create_using= nx.DiGraph if directed else nx.Graph)
    
    if G.number_of_edges() == 0:
        utils_logger.critical(f'Empty graph detected after parsing. It is probably due to an error declaring delimiters or comments in network file -> {file_path}')
        raise NullGraphError(f'Empty graph detected after parsing. It is probably due to an error declaring delimiters or comments in network file -> {file_path}')
    
    return G

# Comparison Fxn
def association(
    dict_data: dict[str, dict[str, float]], corr_func: str | Callable(Iterable, Iterable) = 'pearson',
) -> pd.DataFrame:
    """Computes correlation between elements in a dictionary

    _extended summary_[#_unique ID_]_
    Performs correlation between all elements in a dictionary (pairwise).
    Then calculates a correlation-based distance, for each element, that satisfies the triangle inequality [#OTICBDGEP23].
    Admitted correlations include: pearson correlation, spearman correlation and cosine similarity.

    .. math:: d = \sqrt[]{1 - \left| \rho \right|} [#OTICBDGEP23]
    Where \rho is the correlation coefficient.

    Arguments:
        dict_data (dict[str, dict[str, float]]): dictionary with keys as IDs for each element and values as np.arrays with data.
        corr_func (str | Callable): correlation function desired for analysis. Defaults to 'pearson'.

    Raises:
        ValueError: invalid correlation.
        ValueError: occurs when one or both arrays correlated are constants.

    Returns:
        pd.DataFrame: correlation-based distance matrix for the input data.

    Notes:
        Correlation function must return either a float with the correlation value
        or an Iterable where the first element is the correlation value.

        Correlation function's not optional parameters must only be the two arrays to compare.
    
    References:
        .. [#OTICBDGEP23] *BMC Bioinformatics* 24:40 (2023) https://doi.org/10.1186/s12859-023-05161-y """

    # Correlation fxn
    if corr_func not in CORRELATIONS.keys():
        utils_logger.critical(f"Correlation function not admitted. Admitted options: {CORRELATIONS.keys()}")
        raise ValueError(f"Correlation function not admitted. Admitted options: {CORRELATIONS.keys()}")
    corr_func = CORRELATIONS[corr_func]

    # Get the keys (name_dists) from the dictionary
    name_dists = list(dict_data.keys())

    # Initialize an empty DataFrame to store the correlation coefficients
    corr_df = pd.DataFrame(index=name_dists, columns=name_dists)

    # Calculate the pairwise correlation
    for i in range(len(name_dists)):
        for j in range(i, len(name_dists)):
            name_dist1 = name_dists[i]
            name_dist2 = name_dists[j]
            array1 = np.asarray(list(dict_data[name_dist1].values()))
            array2 = np.asarray(list(dict_data[name_dist2].values()))
            try:
                mask = np.isfinite(array1) & np.isfinite(array2)
            except ValueError:
                utils_logger.critical(f"Error in {name_dist1} or {name_dist2}")
                print(f"array1: {dict_data[name_dist1]}")
                print(f"array2: {dict_data[name_dist2]}")
                raise ValueError
            filtered_array1 = array1[mask]
            filtered_array2 = array2[mask]

            # Calculate Pearson correlation coefficient and p-value
            try:
                result = corr_func(filtered_array1, filtered_array2)
            except TypeError:
                utils_logger.critical('Correlation function not accepted.')

            accepted_types = [(float, Iterable), float]

            if not isinstance(result, accepted_types[0]) and not isinstance(result, accepted_types[1]):
                utils_logger.critical(
                    f"Return Type must be {accepted_types}"
                )
                raise TypeError(f"Correlation function not admitted, Return Type must be {accepted_types}")

            if not isinstance(result, float):
                corr_coef = result[0]
            else:
                corr_coef = result

            # Store the correlation coefficient in the DataFrame
            corr_df.loc[name_dist1, name_dist2] = corr_coef
            corr_df.loc[name_dist2, name_dist1] = corr_coef

    # Calculation of distance ref.
    dist_df = abs(1 - abs(corr_df))
    dist_df = round(dist_df.applymap(np.sqrt), 5)

    return dist_df

# Obtaining models abbreviations for filtering
def get_models_abbreviations(avg_random_scalars_array: dict):
    # Gets files extensions (dictionary keys extensions)
    files_prefix = r'Avg_[^_]+_'
    base_abbreviations = {
            'GNP' : 'ER GNP',
            'GNM' : 'ER GNM',
            'KR' : 'Regular',
            'BA-out' : 'BA (out-degree)',
            'BA-in' : 'BA (in-degree)',
            'BA-degree' : 'BA (degree)'
        }
    abbreviations = {}
    for net_id, _ in avg_random_scalars_array.items():
        try:
            model_abbreviation = re.findall(files_prefix, net_id)[0].split('_')[1]
            if model_abbreviation not in abbreviations.keys():
                abbreviations[model_abbreviation] = base_abbreviations[model_abbreviation] if model_abbreviation in base_abbreviations.keys() else model_abbreviation.replace('-', ' ')
        except IndexError:
            # Detected file name that do not correspond to one of model analog
            pass        

    return abbreviations

# Filtering of association df to compare to models fxn
def filter_association_df_for_models(association_mtrx: pd.DataFrame, abbreviations: dict):
    files_prefix = r'Avg_[^_]+_'
    # Final fig names and values for plotting
    filtered_association = {
        v : [] for k, v in abbreviations.items()
    }
    filtered_association['input networks'] = []
    for index_name in association_mtrx.index:
        if index_name.find('Avg') == -1:
            filtered_columns = [
                col
                for col in association_mtrx.columns
                if index_name == re.sub(files_prefix, '', col)
            ]
            row_values = association_mtrx.loc[index_name, filtered_columns]
            # filtered_association['input networks'].append(index_name.replace('.txt', '').replace('_', ' '))
            filtered_association['input networks'].append(index_name)
            for column, value in row_values.items():
                if column != index_name:
                    filtered_association[abbreviations[column.split('_')[1]]].append(value)
            
    return pd.DataFrame({k : v for k, v in filtered_association.items() if v}, index= filtered_association['input networks']).drop('input networks', axis= 1)

def clean_names_association_df(df):
    # Auxiliar fxn to remove potential file extensions and replace _ for blank spaces
    def clean_name(name):
        # Remove file extensions (.txt, .tsv, .csv, etc.)
        name = re.sub(r'\.(txt|tsv|csv|xlsx|xls|json|parquet|adjlist|graphml)$', '', name)
        # Replace _ for blank spaces
        name = name.replace('_', ' ')
        return name

    # Limpiar nombres de columnas
    df.columns = [clean_name(col) for col in df.columns]
    
    
    # Clean row names (if index is an index of names)
    if df.index.dtype == 'object':
        df.index = [clean_name(idx) for idx in df.index]

    return df

def remove_self_loops(G: nx.DiGraph):
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def compute_moments(data: np.ndarray, ddof: int = 1) -> tuple[float, float, float, float]:
    """Computes the four first moments of a distribution.

    Args:
        data (numpy.array): An array containing the data points of the distribution.
        ddof (int, optional): The delta degrees of freedom. The divisor used in calculations is N - ddof.
            where N represents the number of elements. By default ddof is 1 (for sample data).

    Returns:
        A tuple containing the mean, variance, skewness, and kurtosis of the distribution.

    Note:
        Uniform distributions have np.NAN as kurtosis and skewness.
        Nan will be propagated
    """
    warnings.filterwarnings(action='ignore')
    mean = np.nanmean(data)
    variance = np.nanvar(data, ddof=ddof)
    skewness = skew(data, nan_policy="omit")
    kurt = kurtosis(data, nan_policy="omit")
    warnings.resetwarnings()
    return mean, variance, skewness, kurt

def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))

def get_clusters(
    corr_df, clust_num: int = None, threshold: float = 0.7, ch_method: str = "ward", ch_metric: str = "euclidean", map_ids=True, fcluster_kwargs: dict = None
):
    """Get clusters from a correlation matrix.

    Args:
        corr_df (pandas.DataFrame): A correlation matrix.
        clust_num (int, optional): The number of clusters to be obtained. None to automatically obtain the number of clusters.
        ch_method (str, optional): The linkage method to be used. Dafaults to 'ward'.
        ch_metric (str, optional): The distance metric to be used. Defaults to 'euclidean'.
        map_ids (bool, optional): If True, the clusters will be returned as a dictionary. Defaults to True.
        fcluster_kwargs (dict, optional): Keyword arguments to be passed to scipy.cluster.hierarchy.fcluster.
            t=0.5, criterion="distance" will be used if fcluster_kwargs is None.

    Returns:
        A list containing the cluster number for each node.
        If map_ids is True, a dictionary containing the clusters will be returned.

    Note:
        The distance matrix is computed as 1 - |corr_df|
    """
    corr_df = np.abs(corr_df.astype("float"))
    # corr_df = corr_df.fillna(0)
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    dist_mtrx = round(1 - abs(corr_df), 4)
    try:
        square_matrix = squareform(dist_mtrx)
    except ValueError:
        # warnings.warn(f"NaNs found in the correlation matrix. Unable to compute clusters.")
        utils_logger.critical(f"NaNs found in the correlation matrix. Unable to compute clusters.")
        raise ValueError(f"NaNs found in the correlation matrix. Unable to compute clusters.")
    linkage_mtrx = linkage(square_matrix, method=ch_method, metric=ch_metric)
    index = list(corr_df.index)
    if clust_num is not None:
        cluster_vector = fcluster(linkage_mtrx, t= clust_num, criterion= "maxclust")
    elif threshold is not None:
        cluster_vector = fcluster(linkage_mtrx, t= threshold, criterion= "distance")
    else:
        cluster_vector = fcluster(linkage_mtrx, **fcluster_kwargs)
    
    clusters = {i: [] for i in cluster_vector}
    {clusters[cluster_vector[i]].append(index[i]) for i in range(len(cluster_vector))}

    return cluster_vector if not map_ids else clusters

def giant_component(G):
    """Summary."""
    components = nx.weakly_connected_components if G.is_directed() else nx.connected_components
    return G.subgraph(max(components(G), key=len)).copy()

def giant_component_size(G):
    """Summary."""
    components = nx.weakly_connected_components if G.is_directed() else nx.connected_components
    return len(max(components(G), key=len))

def is_iterable(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, str)

def save_prop_dicts(
    array: dict,
    net_id: str,
    type : str,
    output_dir: str = os.getcwd(),
    delimiter: str = "\t",
    cl: str = None,
) -> None:
    """
    Save the structural properties in a file containing the name of the network and the properties.

    Args:
        array (dict): Dictionary with the properties of the networks.
            {network_name: {property_name: property_value}}.
        net_id (str): Name of the network.
        type (str): Type of properties stored in array.
        output_dir (str): Path to the output directory. Defaults to current directory.
        delimiter (str): Delimiter to use in the output file. Defaults to tab.
        cl (str): Command line used to run the script.

    Returns:
        None.
    """

    exts = {",": "csv", "\t": "tsv"}
    ext = exts.get(delimiter, "txt")

    file_p = concat_path(output_dir, f"{net_id}_{type}_props.{ext}")

    if cl is not None:
        with open(file_p, "w") as f:
            f.write(f"{cl}")

    # save scalar props as csv
    df_s = pd.DataFrame.from_dict(array, orient="index").T
    df_s.to_csv(file_p, sep=delimiter, mode= 'a')

def save_figs(
    fig: matplotlib.figure.Figure,
    props : str = None,
    net_id: str = None,
    output_dir: str = os.getcwd(),
    compare: bool = True,
) -> None:
    """
    Save the structural properties in a file containing the name of the network and the properties.

    Args:
        scalar_props (dict): Dictionary with the scalar properties of the networks.
            {network_name: {property_name: property_value}}.
        dist_props (dict): Dictionary with the distribution properties of the networks.
            {network_name: {property_name: property_moments}}.
        output_dir (str): Path to the output directory. Defaults to current directory.
        delimiter (str): Delimiter to use in the output file. Defaults to tab.
        cl (str): Command line used to run the script.

    Returns:
        None.
    """
    if isinstance(fig, matplotlib.figure.Figure):
        if compare:
            file_p = concat_path(output_dir, f"nets_comparison.png")
        else:
            file_p = concat_path(output_dir, f"{net_id}_{props}_props.png")
        
        fig.savefig(fname= file_p, bbox_inches = "tight", dpi= 300)
        matplotlib.pyplot.close('all')

def common_props_dict(networks):
    new = defaultdict(lambda:defaultdict())

    for i, (net_id, props) in enumerate(networks.items()):
        if i == 0:
            common = set(props.keys())
        else:
            common.intersection_update(set(props.keys()))

    new = {
        net_id : {
            prop_name : value 
            for prop_name, value in props.items()
            if prop_name in common
        }
        for net_id, props in networks.items()
    }

    return new

def process_netective_properties_files(results_dir: str, corr_func: str= 'pearson', metric: str= 'euclidean', method: str= 'ward', compare_to_models: bool= False, features: pd.DataFrame= None, data_type: dict= None, title: str= None)-> matplotlib.figure.Figure:
    """Utils fxn for processing output properties files created by Netective.

    By processing, it is understood comparing the input networks using a clustermap.

    Arguments:
        results_dir (str): directory path with output properties files created by Netective.
        corr_func (str): correlation metric to use to correlate properties' arrays between networks. Defaults to 'pearson'.
        metric (str): distance metric to use . Defaults to 'euclidean'.
        method (str): method to use for clustering. Defaults to "ward".
        compare_to_models (bool): whether the heatmap is comparing input networks to model analogs. Defaults to None.
        features (pd.DataFrame): input features dataframe. Defaults to None.
        data_type (dict): data type of each feature. Defaults to None.
        verbose (str): verbosity level of the logger. Defaults to None.
        title (str): title of the heatmap. Defaults to None.

    Returns:
        matplotlib.figure.Figure: figure containing the heatmap."""
    
    scalars_array = {}
    dist_array = {}
    dist_moments_array = {}
    delimiters = {
        '.tsv' : '\t',
        '.csv' : ',',
        '.txt' : ' ',
    }

    # Retrieve properties values for input networks (not model analogs)
    for dir, _, files in os.walk(results_dir):
        for f in files:
            if f.find('_distributions_props') == -1 and f.find('_scalars_props') == -1 and f.find('_moments_props') == -1:
                utils_logger.warning(f'A file not created by Netective detected, cannot be processed. Skipping it: {f}')
                continue
            temp_dict = {}
            
            # We take advantage of the file extension Netective gives to output files
            file_extension = f.split('_props')[1]
            if f.find('scalars') != -1:
                net_name = f.split('_props')[0].split('_scalars')[0].split('.')[0]
                props_type = 'scalars'
            elif f.find('distributions') != -1:
                net_name = f.split('_props')[0].split('_distributions')[0].split('.')[0]
                props_type = 'distributions'
            else:
                net_name = f.split('_props')[0].split('_moments')[0].split('.')[0]
                props_type = 'moments'
            
            # Read Netective props file
            temp_dict[net_name] = pd.read_csv(
                os.path.join(dir, f),
                sep= delimiters[file_extension],
                comment= '#',
                header = 0,
                index_col= 0
            ).to_dict(orient= 'list')
            
            # Update scalars and distributions arrays
            if props_type == 'scalars':
                scalars_array.update(temp_dict)
            elif props_type == 'distributions':
                dist_array.update(temp_dict)
            else:
                dist_moments_array.update(temp_dict)
    
    # Flatten scalars array
    for net_id, prop in scalars_array.items():
        for prop_id, value in prop.items():
            scalars_array[net_id][prop_id] = float(value[0])
    
    # Compute moments for node-level properties
    # dist_moments_array = {}
    for net_id, prop in dist_array.items():
        temp_dict = {}
        temp_dict[net_id] = {}
        for prop_id, distribution in prop.items():
            temp_dict[net_id][prop_id] = list(compute_moments(data= np.array(distribution)))
        dist_moments_array.update(temp_dict)
    
    # Add distributions averages for input networks to scalars array
    for net_id, prop in dist_moments_array.items():
        for prop_id, moments in prop.items():
            scalars_array[net_id][f'Average {prop_id}'] = moments[0]

    # Association DataFrame
    scalars_array = common_props_dict(scalars_array)
    distances_df = association(scalars_array, corr_func)
    if compare_to_models:
        if not any('Avg' in key for key in scalars_array):
            utils_logger.warning(f'compare_to_models param set, but no properties files detected for analog models detected in: {results_dir}')
            compare_to_models = False
        else:
            abbreviations = get_models_abbreviations(scalars_array)
            distances_df = filter_association_df_for_models(distances_df, abbreviations)
    distances_df = clean_names_association_df(distances_df)

    # Comparison Heatmap
    comp_heatmap = create_comp_heatmap(
        distances_df= distances_df,
        metric= metric,
        method= method,
        features= features,
        data_type= data_type,
        compare_to_models= compare_to_models,
        title= title
    )
    
    return comp_heatmap

# Paths objects
class ShortestDistances:
    """Summary."""

    def __init__(self, G):
        """Summary."""
        if G.is_directed():
            utils_logger.critical("requires an undirected graph")
        iG = ig.Graph.TupleList(
            G.edges(data=False),
            directed=False,
            vertex_name_attr="name",
            edge_attrs=None,
            weights=False,
        )
        iG.add_vertices(nx.isolates(G))
        self.__id2name = {i.index: i["name"] for i in iG.vs}
        self.__name2id = {v: iG.vs.find(name=v).index for v in G.nodes()}
        self.__sp = np.ma.masked_values(np.asarray(iG.distances(), dtype=np.float64), np.inf)
        self.__num_nodes = self.__sp.shape[0]

    def matrix(self, masked_inf=False):
        """Summary."""
        return self.__sp if masked_inf else self.__sp.filled()

    @property
    def average_path_length(self):
        """Summary."""
        n = self.__num_nodes
        return self.__sp.sum() / (n * (n - 1))

    @property
    def diameter(self):
        """Summary."""
        return int(self.__sp.max())

    @property
    def radius(self):
        """Summary."""
        return min(self.eccentricity().values())

    @property
    def center(self):
        """Summary."""
        r = self.radius
        return {v for v, e in self.eccentricity().items() if e == r}

    @property
    def periphery(self):
        """Summary."""
        d = self.diameter
        return {v for v, e in self.eccentricity().items() if e == d}

    def eccentricity(self, v=None):
        """Summary."""
        if v is None:
            v = range(self.__num_nodes)
        else:
            v = [self.__name2id[i] for i in v] if is_iterable(v) else self.__name2id[v]
        if is_iterable(v):
            return {self.__id2name[i]: int(self.__sp[i, :].max()) for i in v}
        else:
            return int(self.__sp[v, :].max())

    def shortest_path_length(self, v, u=None):
        """Summary."""
        if u is None:
            return self.__sp.filled()[self.__name2id[v], :]
        else:
            if not is_iterable(u):
                return self.__sp.filled()[self.__name2id[v], self.__name2id[u]]
            else:
                return self.__sp.filled()[self.__name2id[v], [self.__name2id[i] for i in u]]

class ShortestPaths:
    """Summary."""

    def __init__(self, G):
        """Summary."""
        if G.is_directed():
            utils_logger.critical("Requires an undirected graph")
            raise TypeError('Requires an undirected graph')
        self.__G = ig.Graph.TupleList(
            G.edges(data=False),
            directed=False,
            vertex_name_attr="name",
            edge_attrs=None,
            weights=False,
        )
        self.__G.add_vertices(nx.isolates(G))
        self.__id2name = {i.index: i["name"] for i in self.__G.vs}
        self.__name2id = {v: self.__G.vs.find(name=v).index for v in G.nodes()}

    # def shortest_paths(self, v, u=None):
    #     """Summary."""
    #     if u is not None:
    #         u = [self.__name2id[i] for i in u] if is_iterable(u) else self.__name2id[u]
    #     paths = self.__G.get_all_shortest_paths(self.__name2id[v], to=u, weights=None, mode="out")
    #     return tuple((tuple((self.__id2name[v] for v in p)) for p in paths if len(p) > 1))

    def betweenness(self, vertices=None, cutoff=None, sources=None, targets=None):
        """Summary."""
        if vertices is not None:
            vertices = (
                [self.__name2id[i] for i in vertices]
                if is_iterable(vertices)
                else self.__name2id[vertices]
            )
        if sources is not None:
            sources = (
                [self.__name2id[i] for i in sources]
                if is_iterable(sources)
                else self.__name2id[sources]
            )
        if targets is not None:
            targets = (
                [self.__name2id[i] for i in targets]
                if is_iterable(targets)
                else self.__name2id[targets]
            )
        betweenness = self.__G.betweenness(
            vertices=vertices,
            directed=False,
            cutoff=cutoff,
            weights=None,
            sources=sources,
            targets=targets,
        )
        return (
            {v: b for v, b in zip(self.__G.vs["name"], betweenness)}
            if is_iterable(vertices)
            else betweenness
        )

# Effciciency object
class Efficiency:
    """
    Reference:
        Vito Latora and Massimo Marchiori. Efficient behavior of small-world networks. *Physical Review Letters* 87.19 (2001): 198701. http://dx.doi.org/10.1103/PhysRevLett.87.198701
    """

    def __init__(self, G, shortest_distances=None):
        """Summary."""
        if G.is_directed():
            utils_logger.critical("Efficiency is not defined for directed graphs")
            raise TypeError("Efficiency is not defined for directed graphs")
        if shortest_distances is None:
            sp = ShortestDistances(G)
        else:
            if not isinstance(shortest_distances, ShortestDistances):
                raise TypeError("shortest_distances must be a ShortestDistance instance")
            sp = shortest_distances
        self.__name2id = sp._ShortestDistances__name2id
        self.__efficiency = 1 / sp.matrix(masked_inf=True)
        self.__efficiency = self.__efficiency.filled(fill_value=0)
        self.__G = G
        self.__num_nodes = self.__efficiency.shape[0]

    def efficiency(self, u, v):
        """Summary."""
        return self.__efficiency[self.__name2id[u], self.__name2id[v]]

    @property
    def global_efficiency(self):
        """Summary."""
        n = self.__num_nodes
        if n > 1:
            return self.__efficiency.sum() / (n * (n - 1))
        else:
            return 0

    @property
    def local_efficiency(self):
        """Summary."""
        lg = (tuple(self.__G.neighbors(v)) for v in self.__G.nodes())
        ge = []
        for g in lg:
            if len(g) < 2:
                continue
            ef = ShortestDistances(self.__G.subgraph(g))
            ef = 1 / ef.matrix(masked_inf=True)
            ef = ef.filled(fill_value=0)
            nl = ef.shape[0]
            ge.append(ef.sum() / (nl * (nl - 1)))
        return np.array(ge).sum() / self.__num_nodes

# motifs class
class count_3motifs:
    """Summary."""

    def __init__(self, G):
        if not G.is_directed():
            utils_logger.critical("requires a directed graph")
        iG = ig.Graph.TupleList(
            G.edges(data=False),
            directed=True,
            vertex_name_attr="name",
            edge_attrs=None,
            weights=False,
        )
        iG.add_vertices(nx.isolates(G))
        self.tc = iG.triad_census()

    @property
    def feedforwards(self):
        """Summary."""
        return self.tc.t030T

    @property
    def complex_feedforwards(self):
        """Summary."""
        return self.tc.t120U

    @property
    def feedbacks(self):
        """Summary."""
        return self.tc.t030C + self.tc.t120C + self.tc.t210 + 2 * self.tc.t300
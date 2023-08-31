from __future__ import annotations

"""Utility functions for the netective package."""

__all__ = [
    "concat_path",
    "run_parallel",
    "validate_network",
    "parse_nets",
    "flatten_list_of_iterables",
    "compute_moments",
]

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from tqdm import tqdm
import concurrent.futures
from itertools import chain
from scipy.stats import pearsonr
from collections import defaultdict
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Union, Callable, Iterable

concat_path = os.path.join


def run_parallel(f, my_iter, workers):

    """
    Start the parallel processes.

    Parameters
    ----------
    f: function.
        Function to be executed in parallel.
    my_iter: Iterable.
        Iterable with the inputs for f.
        Each element of iterable will be unzipped before calling f.
    workers: Numer of processes to run in parallel.

    Returns
    -------
    Results: zip object.
        Contains the results of the function f.
    """

    len_iter = len(my_iter)
    with tqdm(total=len_iter) as pbar:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for arg in zip(*my_iter):
                    name = arg[1]
                    print(f"Running {name}")
                    futures[executor.submit(f, *arg)] = name

                results = defaultdict(dict)
                for future in concurrent.futures.as_completed(futures):
                    try:
                        scalar, dist = future.result()
                        results["scalars"].update(scalar)
                        results["distributions"].update(dist)
                        pbar.update(1)
                    except Exception as exc:
                        print(f"Error: {exc}")
        except NotImplementedError as e:
            print()
            print(e.message)

    return results


def validate_network(G: nx.DiGraph | nx.Graph) -> Union(nx.DiGraph, nx.Graph):
    """Validates the network and returns a DiGraph or Graph object."""
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError("G must be a DiGraph or a Graph")
    if G.size() == 0:
        raise ValueError(f"G must have at least one edge. It has {G.size()} edges.")
    return G


def parse_nets(paths: list[str], comments: str = "#", delimiter: str = "\t") -> dict:

    """Reads network files and returns a dictionary of networkx.DiGraphs.

    Firts column of the network file is considered as the source node and the
    second column is considered as the target node. The network file must be
    delimited by a tab character.

    Args:
        paths (list[str]): List of paths to network files.
        comments (str, optional): Comment character. Defaults to '#'.
        delimiter (str, optional): Delimiter character. Defaults to '\t'.

    Returns:
        dict: Dictionary of networkx.DiGraphs.

    Raises:
        ValueError: If the network file is not a DiGraph.

    TODO:
        * Add support for metadata (scores of the predictions).
        * Raise Error when len(tfs & tgs) = 0.
        * Allow to cut the network by the number of edges. (first n edges).
    """

    networks = {}

    for net_path in paths:

        net_name = os.path.basename(net_path)

        # read network file (only DiGraphs with no metadata are supported)
        networks[net_name] = nx.read_edgelist(
            net_path,
            comments=comments,
            delimiter=delimiter,
            create_using=nx.DiGraph,
            data=False
            # encoding='utf-8'
        )

    return networks


def parse_network(
    file_path, comments="#", delimiter="\t", directed=True, score=False, use_position_as_score=False
) -> Union(nx.DiGraph, nx.Graph):
    """
    Parse a network file and return a networkx.DiGraph or networkx.Graph depending on the directed parameter.

    Args:
        file_path: Path to the network file.
        comments: Comment character.
        delimiter: Delimiter character.
        directed: If True, the network will be a DiGraph, otherwise it will be a Graph.
        score: If True, the network will use the third column of the file as the score of the edge.
        use_position_as_score: If True, the position of the edge in the file will be used as the score of the edge.
    """
    if score and use_position_as_score:
        raise ValueError("score and use_position_as_score cannot be True at the same time.")

    if not use_position_as_score:
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
    return G

# Comparison Fxn
def association(
    dict_data: dict[str, dict[str, float]], corr_func: Callable(Iterable, Iterable) = pearsonr
) -> pd.DataFrame:
    """
    Computes correlation between elements in a dictionary

    Args:
        dict_data : dictionary with keys as IDs for each element and values as np.arrays with data.
        corr_func : correlation function desired for analysis. Default is pearsonr from scipy.

    Returns:
        corr_df : DataFrame with the correlation results of the input data.

    Note:
        Correlation function must return either a float with the correlation value
        or an Iterable where the first element is the correlation value.

        Correlation function's not optional parameters must only be the two arrays to compare.

        All scipy.stats functions admitted except page_trend_test
    """
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

            mask = np.isfinite(array1) & np.isfinite(array2)
            filtered_array1 = array1[mask]
            filtered_array2 = array2[mask]

            # Calculate Pearson correlation coefficient and p-value
            result = corr_func(filtered_array1, filtered_array2)

            accepted_types = (float, Iterable)

            if not isinstance(result, accepted_types):
                raise TypeError(
                    f"Correlation function not admitted, Return Type must be {accepted_types}"
                )

            if not isinstance(result, float):
                corr_coef = result[0]
            else:
                corr_coef = result

            # Store the correlation coefficient in the DataFrame
            corr_df.loc[name_dist1, name_dist2] = corr_coef
            corr_df.loc[name_dist2, name_dist1] = corr_coef

    return corr_df

def remove_self_loops(G: nx.DiGraph):
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def compute_moments(data: np.ndarray, ddof: int = 1) -> tuple[float, float, float, float]:
    """Computes the four first moments of a distribution.

    Args:
        data: An array containing the data points of the distribution.
        ddof: The delta degrees of freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is 1 (for sample data).

    Returns:
        A tuple containing the mean, variance, skewness, and kurtosis of the distribution.

    Note:
        Uniform distributions have np.NAN as kurtosis and skewness.
        Nan will be propagated
    """
    mean = np.nanmean(data)
    variance = np.nanvar(data, ddof=ddof)
    skewness = skew(data, nan_policy="omit")
    kurt = kurtosis(data, nan_policy="omit")

    return mean, variance, skewness, kurt


def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))


def get_clusters(
    corr_df, clust_num, ch_method: str = "ward", ch_metric: str = "euclidean", map_ids=True
):
    """Get clusters from a correlation matrix.

    Args:
        corr_df: A correlation matrix.
        clust_num: The number of clusters to be obtained.
        ch_method: The linkage method to be used.
        ch_metric: The distance metric to be used.
        map_ids: If True, the clusters will be returned as a dictionary.

    Returns:
        A list containing the cluster number for each node.
        If map_ids is True, a dictionary containing the clusters will be returned.

    Note:
        The distance matrix is computed as 1 - |corr_df|
    """
    corr_df = np.abs(corr_df.astype("float"))
    # corr_df = corr_df.fillna(0)
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    dist_mtrx = round(1 - corr_df, 4)
    try:
        square_matrix = squareform(dist_mtrx)
    except ValueError:
        # warnings.warn(f"NaNs found in the correlation matrix. Unable to compute clusters.")
        raise ValueError(f"NaNs found in the correlation matrix. Unable to compute clusters.")
    linkage_mtrx = linkage(square_matrix, method=ch_method, metric=ch_metric)
    index = list(corr_df.index)
    cluster_vector = fcluster(linkage_mtrx, t=clust_num, criterion="maxclust")
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


# Paths objects


def is_iterable(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, str)


class ShortestDistances:
    """Summary."""

    def __init__(self, G):
        """Summary."""
        if G.is_directed():
            raise TypeError("requires an undirected graph")
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
            raise TypeError("requires an undirected graph")
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

    def shortest_paths(self, v, u=None):
        """Summary."""
        if u is not None:
            u = [self.__name2id[i] for i in u] if is_iterable(u) else self.__name2id[u]
        paths = self.__G.get_all_shortest_paths(self.__name2id[v], to=u, weights=None, mode="out")
        return tuple((tuple((self.__id2name[v] for v in p)) for p in paths if len(p) > 1))

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
            raise TypeError("efficiency is not defined for directed graphs")
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

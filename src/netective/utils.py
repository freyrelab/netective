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
import numpy as np
import networkx as nx
from tqdm import tqdm
import concurrent.futures
from itertools import chain
from collections import defaultdict
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from freyrelab.regnets import regnet as rn

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


def validate_network(G: nx.DiGraph | rn.RegNet) -> rn.RegNet:
    """Validates the network and returns a RegNet object."""
    print(f"validate --- G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    if isinstance(G, nx.DiGraph):
        G = rn.RegNet(G)
        print(
            f"validate RegNet --- G has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
        )
    elif not isinstance(G, rn.RegNet):
        raise TypeError("G must be a DiGraph or a RegNet")
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
    dist_mtrx = round(1 - np.abs(corr_df.astype("float")), 4)
    linkage_mtrx = linkage(squareform(dist_mtrx), method=ch_method, metric=ch_metric)
    index = list(corr_df.index)
    cluster_vector = fcluster(linkage_mtrx, t=clust_num, criterion="maxclust")
    clusters = {i: [] for i in cluster_vector}
    {clusters[cluster_vector[i]].append(index[i]) for i in range(len(cluster_vector))}

    return cluster_vector if not map_ids else clusters

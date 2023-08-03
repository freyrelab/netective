# Imports
from __future__ import annotations

import inspect
import traceback
from typing import Tuple
from warnings import warn
from itertools import chain
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from freyrelab.regnets.regnet import RegNet
from freyrelab.nets.paths2 import Efficiency, ShortestDistances, ShortestPaths

from netective import properties
from netective.properties import remove_self_loops
from netective.utils import compute_moments, run_parallel
from netective.structure.dataviz import plot_scalars, create_symmetric_heatmap, plot_distributions


# User Fxns
# Characterization of one network
def characterize_network(
    G: RegNet, name: str, norm: str | None = None, selected_props: str | list = "all"
) -> None:
    """Module-level function to characterize a single network.

    Args:
        G (RegNet): Network to characterize.
        norm (str, optional): Normalization to apply. Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).

    Raises:
        Exception: Raised if the normalization is not valid.
    """
    if norm not in NORM_OPTIONS:
        raise properties.NormalizationError("Normalization not valid")

    scalar_values, dist_values = get_props(G, name, norm, props=selected_props)

    if len(dist_values) == 0 and len(scalar_values) == 0:
        raise ValueError("Not enough data, try with more properties or another normalization")

    if len(dist_values) != 0:
        fig_dist, _ = plot_distributions(dist_values)
        fig_dist.show()
    if len(scalar_values) != 0:
        fig_scalar, _ = plot_scalars(scalar_values)
        fig_scalar.show()


# Comparison of multiple networks
def compare_networks(
    networks: dict,
    norm: str | None = None,
    selected_props: str | list = "all",
    workers: str | int = "auto",
) -> Tuple[plt.figure.Figure, plt.figure.Figure]:

    """Module-level function to compare multiple networks.

    Args:
        networks (dict): Dictionary of networks to compare {'name':RegNet}.
        norm (str, optional): Normalization to apply. Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
        workers (int, optional): Number of workers to use. Defaults to 'auto'.

    Raises:
        NormalizationError: Raised if the normalization is not valid.
        ValueError: Raised if there is not enough data to compare.
    """

    if norm not in NORM_OPTIONS:
        raise properties.NormalizationError("Normalization not valid")

    # handle workers
    usable_workers = cpu_count() - 1
    if workers == "auto":
        workers = usable_workers
    elif workers > usable_workers:
        warn(
            f"{workers} workers requested, but only {usable_workers} are available. Using {usable_workers} workers instead."
        )
        workers = usable_workers

    # currently, both selected_props and child_classes are being passed to get_props, however, only one is needed.
    # passing only child_classes would be more efficient beacuse it computes get_child_classes only once.
    child_classes = get_child_classes(PARENT_CLASS, selected_props)

    # prepare data
    data = [
        list(networks.values()),
        list(networks.keys()),
        [norm] * len(networks),
        [selected_props] * len(networks),
        [child_classes] * len(networks),
    ]

    # run parallel
    results = run_parallel(get_props, data, workers)
    name_scalars_array = results["scalars"]
    name_moments_arrays = results["distributions"]

    # Scalar properties
    if len(name_scalars_array) > 0 and len(list(name_scalars_array.values())[0]) > 1:
        df = pairwise_pearson_correlation(name_scalars_array)
        fig_scalar, _ = create_symmetric_heatmap(df, title=f"Global properties")
    else:
        raise ValueError("Not enough data to compare.")

    # Distribution properties
    if len(name_moments_arrays) > 0 and len(list(name_moments_arrays.values())[0]) > 1:
        df = pairwise_pearson_correlation(name_moments_arrays)
        fig_dist, _ = create_symmetric_heatmap(df, title=f"Local properties")
    else:
        raise ValueError("Not enough data to compare.")

    return fig_scalar, fig_dist

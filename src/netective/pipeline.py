# Imports
from __future__ import annotations
import os
import pandas as pd
from io import BytesIO
import networkx as nx
from PIL import Image
import inspect
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import chain

import freyrelab
from freyrelab.regnets.regnet import RegNet
from freyrelab.nets.paths2 import ShortestPaths, Efficiency, ShortestDistances

from netective import properties_2
from netective.utils import compute_moments

# Auxiliar Fxns
def remove_self_loops(G: nx.DiGraph):
    print(type(G))
    G.remove_edges_from(nx.selfloop_edges(G))

def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))

# Get properties selected Fxn
def get_child_classes(parent_class, properties):
    child_classes = []
    all_properties = []
    print(f'Properties used for analysis: ', end=' ')
    if properties == 'all':
        for name, obj in inspect.getmembers(properties_2):
            if (
                inspect.isclass(obj)
                and issubclass(obj, parent_class)
                and obj != parent_class
            ):
                print(obj.CLASS_NAME, end=' ')
                child_classes.append(obj)
                all_properties.append(obj.CLASS_NAME)
    else:
        for name, obj in inspect.getmembers(properties_2):
            if (
                inspect.isclass(obj)
                and issubclass(obj, parent_class)
                and obj != parent_class
                and name in properties
            ):
                print(name, end=' ')
                child_classes.append(obj)
                all_properties.append(obj.CLASS_NAME)
            if (
                inspect.isclass(obj)
                and issubclass(obj, parent_class)
                and obj != parent_class
                and obj.CLASS_NAME not in all_properties
            ):
                all_properties.append(obj.CLASS_NAME)
    print('\n')
    if len(child_classes) == 0:
        raise Exception(f'Sorry, no matches for properties inquired.\nList of available properties is: {all_properties}')
    return child_classes

# Get Instances Fxns
def get_instances_no_paths(G, child_classes):
    instances = {x.CLASS_NAME: x(G) for x in child_classes if not x._use_paths}
    return instances

def get_instances_paths(G, child_classes, net_shortest_paths, net_shortest_distances):
    instances = {x.CLASS_NAME: x(G, net_shortest_paths = net_shortest_paths, net_shortest_distances = net_shortest_distances) for x in child_classes if x._use_paths}
    return instances

# Get Properties and Normalize Values Fxns
def normalize_props(instances, G, norm):
    norm_scalar_values = {}
    norm_dist_values = {}
    for name, x in instances.items():
        dict_ = (
            norm_scalar_values
            if x._return_type == "scalar"
            else norm_dist_values
        )
        try:
            if norm == "network":
                dict_[x.CLASS_NAME] = x.norm_network()
            elif norm == "biological":
                dict_[x.CLASS_NAME] = x.norm_biol()
        except (NotImplementedError, properties_2.NormalizationError):
            dict_[x.CLASS_NAME] = np.nan

    return norm_scalar_values, norm_dist_values

def get_props(G, norm, child_classes):
    # Properties that do not use paths object
    instances = get_instances_no_paths(G, child_classes)
    
    # Paths objects
    remove_self_loops(G)
    G = G.giant_component
    G = G.to_undirected()
    net_shortest_paths = ShortestPaths(G)
    net_shortest_distances = ShortestDistances(G)
    dist_values = {}
    
    # Properties that use paths object
    # They use the giant component from an undirected graph with no selfloops
    instances.update(get_instances_paths(G, child_classes, net_shortest_paths, net_shortest_distances))
    
    scalar_values = {
        x.CLASS_NAME : x.compute()
        for name, x in instances.items()
        if x._return_type == 'scalar'
    }

    dist_values = {
        x.CLASS_NAME : x.compute()
        for name, x in instances.items()
        if x._return_type == 'distribution'
    }

    if norm is not None:
        scalar_values, dist_values = normalize_props(instances, G, norm=norm)
    
    dist_values = {
        k: v for k, v in dist_values.items() if not np.isnan(v).all()
    }

    return scalar_values, dist_values

def plot_distributions(dist_values):
    # Determine the grid shape based on the number of items
    num_items = len(dist_values)
    if num_items > 1: 
        grid_shape = (
            int(np.sqrt(num_items)) + 1,
            int(np.ceil(np.sqrt(num_items))) + 1,
        )
    else:
        grid_shape = (1,1)

    # Create the figure and subplots
    fig, axs = plt.subplots(
        nrows=grid_shape[0],
        ncols=grid_shape[1],
        figsize=(3 * grid_shape[0], 1.5 * grid_shape[1]),
    )

    # Flatten the axes array if it's more than 1D
    if num_items > 1:
        axs = axs.flatten()

    # Iterate over the dictionary items and create the subplots
    for i, (title, data) in enumerate(dist_values.items()):
        ax = (
            axs[i] if num_items > 1 else axs
        )  # Use a single axis if there's only one item
        sns.kdeplot(data, ax=ax, fill=True, color="#384265")
        ax.set_title(title)

    # Remove any extra empty subplots, only if there is more than one distribution to plot
    if num_items > 1:
        if num_items < axs.size:
            for j in range(num_items, axs.size):
                fig.delaxes(axs[j])
            # Adjust spacing between subplots
            fig.tight_layout()
    return fig, axs


def plot_scalars(data_dict):
    # Extract keys (strings) and values (floats) from the dictionary
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    with sns.axes_style("darkgrid"):
        # Create the figure and axes
        fig, axs = plt.subplots(figsize=(2, 0.3 * len(labels)))

        # Create a horizontal bar plot using seaborn
        sns.barplot(x=values, y=labels, ax=axs, color="#384265")

        # Add annotations to the bars
        for i, value in enumerate(values):
            if isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    value = round(value, 2) if value >= 0.01 else value
            plt.text(
                value,
                i,
                str(value) if value >= 0.01 else f"{value:.2E}",
                va="center",
            )

        # Customize the plot
        axs.set_xlabel("Values")
        axs.set_ylabel("")
        axs.set_title('Network Level Properties', loc='left')

    return fig, axs

# Plotting Fxns
def create_symmetric_heatmap(dataframe, title: str):
    # Create a figure and axes
    # fig, axs = plt.subplots()

    # Plot the heatmap
    g = sns.clustermap(
        dataframe.astype(float),
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar=True,
    )

    # Set the title
    g.ax_heatmap.set_title(title)

    # Return the figure and axes
    return g.fig, g.ax_heatmap

# Comparison Fxn
def pairwise_pearson_correlation(
    dict_data,
):  # TODO: make this general for any correlation... # shouldn't be here...
    # Get the keys (name_dists) from the dictionary
    name_dists = list(dict_data.keys())

    # Initialize an empty DataFrame to store the correlation coefficients
    corr_df = pd.DataFrame(index=name_dists, columns=name_dists)

    # Calculate the pairwise Pearson correlation coefficients
    for i in range(len(name_dists)):
        for j in range(i, len(name_dists)):
            name_dist1 = name_dists[i]
            name_dist2 = name_dists[j]
            array1 = dict_data[name_dist1]
            array2 = dict_data[name_dist2]

            mask = np.isfinite(array1) & np.isfinite(array2)
            filtered_array1 = array1[mask]
            filtered_array2 = array2[mask]

            # Calculate Pearson correlation coefficient and p-value
            corr_coef, _ = pearsonr(
                filtered_array1, filtered_array2
            )  # TODO!!! genera error por nan o inf. Debes filtrar antes de calcular la correlación, en todos los vectores a comparar.

            # Store the correlation coefficient in the DataFrame
            corr_df.loc[name_dist1, name_dist2] = corr_coef
            corr_df.loc[name_dist2, name_dist1] = corr_coef

    return corr_df

#Main Fxns

    # Analyzing one network
def main_single(G, norm, child_classes):
    scalar_values, dist_values = get_props(G, norm, child_classes)
    if len(dist_values) != 0:
        fig_dist, _ = plot_distributions(dist_values)
        fig_dist.show()
    if len(scalar_values) != 0:
        fig_scalar, _ = plot_scalars(scalar_values)
        fig_scalar.show()

    # Analyzing more than one network
def main_multiple(networks, norm, properties):
    # Iterate over the uploaded files and perform the necessary operations
    for name, G in networks.items():
        scalar_arrays, dist_moments_arrays = compute_props(G, name, norm, properties)

    # Scalar properties
    if len(scalar_arrays) > 0:
        df = pairwise_pearson_correlation(scalar_arrays)
        fig_scalar, _ = create_symmetric_heatmap(
            df, title=f"Network-level properties"
        )
        fig_scalar.show()
    
    #Distribution properties
    if len(dist_moments_arrays) > 0:
        df = pairwise_pearson_correlation(dist_moments_arrays)
        fig_dist, _ = create_symmetric_heatmap(df, title=f"Node-level properties")
        fig_dist.show()

def compute_props(G, name, norm, properties):
    scalar_arrays = {}
    dist_moments_arrays = {}

    parent_class = properties_2._Property
    child_classes = get_child_classes(parent_class, properties)

    # props
    scalar_values, dist_values = get_props(G, norm, child_classes)
    # scalar properties
    scalar_arrays[name] = np.asarray(
        list(scalar_values.values())
    )
    dist_moments = [
        compute_moments(array) for array in dist_values.values()
    ]
    dist_moments_arrays[name] = np.asarray(
        flatten_list_of_iterables(dist_moments)
    )
    return scalar_arrays, dist_moments_arrays

# User Fxns
    # Characterization of one network
def characterization(G: RegNet, norm=None, properties=['AverageClusteringCoefficient', 'AverageDegreeNearestNeighbors', 'AverageLocalEfficiency', 'AverageOutDegreeNearestNeighbors', 'AverageShortestPathLength', 'BetweennessCentrality', 'Center', 'ClusteringCoefficient', 'ComplexFeedForwardCircuits', 'Density', 'Diameter', 'Eccentricity', 'EntropyPKout', 'FeedbackLoops_3', 'GenesintheGiantComponent', 'GiniIndex', 'GlobalEfficiency', 'InDegree', 'LocalityIndex', 'MaxInDegree', 'MaxOutDegree', 'OutDegree', 'Periphery', 'Radius', 'Regulators', 'RichClub', 'SelfRegulations', 'SubgraphCentrality'],):
    if norm == 'network' or norm == 'biological' or norm is None:
        parent_class = properties_2._Property
        selected_child_classes = get_child_classes(parent_class, properties)
        main_single(G, norm, selected_child_classes)
    else:
        raise Exception('Normalization not valid')
    
    # Comparison of multiple networks
def comparison(networks: dict, norm:None=None, properties: str | list='all') -> None:
    if norm == 'network' or norm == 'biological' or norm is None:
        main_multiple(networks, norm, properties)
    else:
        raise Exception('Normalization not valid')


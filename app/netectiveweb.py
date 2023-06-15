import os
import streamlit as st
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


from netective import properties

# from netective.structure import Structure
from netective.utils import compute_moments

# from netective.utils import parse_nets, flatten_list_of_iterables
from freyrelab.regnets import regnet as rn

# Set up the user interface
print("----------------------------path", os.getcwd())
image = Image.open(r"./assets/on_white.png")
width, height = image.size
st.image(image, width=int(width * 0.15))
st.title("Compute network structural properties")
uploaded_files = st.file_uploader(
    "Choose up to three files", accept_multiple_files=True
)

parent_class = properties._Property


def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))


def get_child_classes(parent_class):
    child_classes = []
    for name, obj in inspect.getmembers(properties):
        if (
            inspect.isclass(obj)
            and issubclass(obj, parent_class)
            and obj != parent_class
        ):
            child_classes.append(obj)
    return child_classes


child_classes = get_child_classes(parent_class)


def get_instances(G, child_classes):
    instances = {x.__name__: x(G) for x in child_classes}
    return instances


def normalize_props(instances, G, norm="network"):
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
                dict_[name] = x.norm_network()
            elif norm == "biological":
                dict_[name] = x.norm_biol()
        except (NotImplementedError, properties.NormalizationError):
            dict_[name] = np.nan

    return norm_scalar_values, norm_dist_values


def plot_distributions(dist_values):

    # Determine the grid shape based on the number of items
    num_items = len(dist_values)
    grid_shape = (
        int(np.sqrt(num_items)) + 1,
        int(np.ceil(np.sqrt(num_items))) + 1,
    )

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
        print(i, title, len(axs))
        ax = (
            axs[i] if num_items > 1 else axs
        )  # Use a single axis if there's only one item
        sns.kdeplot(data, ax=ax, fill=True, color="#384265")
        ax.set_title(title)

    # Remove any extra empty subplots
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

    return fig, axs


def get_props(G, norm="network"):
    instances = get_instances(G, child_classes)
    scalar_values = {
        name: x.compute()
        for name, x in instances.items()
        if x._return_type == "scalar"
    }
    dist_values = {
        name: x.compute()
        for name, x in instances.items()
        if x._return_type == "distribution"
    }

    if norm:
        scalar_values, dist_values = normalize_props(instances, G, norm=norm)

    dist_values = {
        k: v for k, v in dist_values.items() if not np.isnan(v).all()
    }
    scalar_values = {k: v for k, v in scalar_values.items() if not np.isnan(v)}

    return scalar_values, dist_values


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

            # Calculate Pearson correlation coefficient and p-value
            corr_coef, _ = pearsonr(array1, array2)

            # Store the correlation coefficient in the DataFrame
            corr_df.loc[name_dist1, name_dist2] = corr_coef
            corr_df.loc[name_dist2, name_dist1] = corr_coef

    return corr_df


# def create_symmetric_heatmap(dataframe, title:str):
#     # Create a figure and axes
#     fig, axs = plt.subplots()

#     # Create a masked array to hide the upper triangle
#     mask = pd.DataFrame(np.triu(np.ones_like(dataframe)), index=dataframe.index, columns=dataframe.columns)

#     # Plot the heatmap
#     sns.heatmap(dataframe.astype(float), cmap='Blues', vmin=0, vmax=1, mask=mask, annot=True, fmt=".2f", cbar=True, ax=axs)

#     # Set the title
#     axs.set_title(title)

#     # Return the figure and axes
#     return fig, axs


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


def main_single(G, norm="network"):
    scalar_values, dist_values = get_props(G, norm=norm)

    fig_dist, _ = plot_distributions(dist_values)
    fig_scalar, _ = plot_scalars(scalar_values)

    # Create two columns for plots
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig_scalar)  # , use_container_width=False)
    with col2:
        st.pyplot(fig_dist)


def main_multiple(uploaded_files, norm):
    # Iterate over the uploaded files and perform the necessary operations
    scalar_arrays = {}
    dist_moments_arrays = {}
    for i, uploaded_file in enumerate(uploaded_files):
        st.write(f"Processing file {i+1}: {uploaded_file.name}")
        file_bytes = uploaded_file.read()
        file_obj = BytesIO(file_bytes)
        G = nx.read_edgelist(
            file_obj, delimiter=" ", create_using=nx.DiGraph, data=False
        )
        G = rn.RegNet(G)
        # props
        scalar_values, dist_values = get_props(G, norm=norm)
        # scalar properties
        scalar_arrays[uploaded_file.name] = np.asarray(
            list(scalar_values.values())
        )
        dist_moments = [
            compute_moments(array) for array in dist_values.values()
        ]
        dist_moments_arrays[uploaded_file.name] = np.asarray(
            flatten_list_of_iterables(dist_moments)
        )
        print(dist_moments_arrays[uploaded_file.name])

    df = pairwise_pearson_correlation(scalar_arrays)
    st.dataframe(df)
    fig_scalar, _ = create_symmetric_heatmap(
        df, title=f"Network-level properties"
    )
    # distribution properties
    df = pairwise_pearson_correlation(dist_moments_arrays)
    st.dataframe(df)
    fig_dist, _ = create_symmetric_heatmap(df, title=f"Node-level properties")

    # Create two columns for plots
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig_scalar)  # , use_container_width=False)
    with col2:
        st.pyplot(fig_dist)


# Define the backend functionality
norm_options = [None, "biological", "network"]
norm = st.selectbox("Normalization", norm_options)

if uploaded_files:
    num_files = len(uploaded_files)
    if num_files == 1:
        # Process for a single file
        file_bytes = uploaded_files[0].read()
        file_obj = BytesIO(file_bytes)
        G = nx.read_edgelist(
            file_obj, delimiter=" ", create_using=nx.DiGraph, data=False
        )
        G = rn.RegNet(G)
        main_single(G, norm=norm)
    elif num_files <= 3:
        # Process for multiple files
        main_multiple(uploaded_files, norm=norm)
    else:
        # Show an error message if more than 3 files are uploaded
        st.error("You can upload up to three files.")

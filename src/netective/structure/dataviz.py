from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from netective.logging_info import get_logger, set_log_level
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import logging
import math

CATEGORICAL = sns.color_palette("Paired") # for categorical data
NUMERICAL = "rocket_r"  #  for numerical data

def ceil_to_next_power_of_10(number):
    return 10 ** math.ceil(math.log10(number))

dataviz_logger = get_logger(__name__)

def format_title(input_str):
    # Check if the length of the string is greater than 22
    if len(input_str) > 22:

        words = input_str.split()

        result = []
        current_line = words[0]

        # Iterate through the words
        for word in words[1:]:
            # Check if the word fits in the current line
            if len(current_line) + len(word) + 1 <= 22:
                current_line += ' ' + word
            else:
                result.append(current_line)
                current_line = word

        # Add the last line to the result
        result.append(current_line)

        formatted_result = '\n'.join(result)

        return formatted_result

    else:
        return input_str



def plot_distributions(dist_values, verbose: str = None):
    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    dataviz_logger.info('Plotting node-level properties...')
    # Determine the grid shape based on the number of items
    num_items = len(dist_values)
    if num_items == 0:
        return None, None
    if num_items > 1:
        grid_shape = (int(np.sqrt(num_items)) + 1, int(np.ceil(np.sqrt(num_items))) + 1)
    else:
        grid_shape = (1, 1)

    # Create the figure and subplots
    fig, axs = plt.subplots(
        nrows=grid_shape[0], ncols=grid_shape[1], figsize=(3 * grid_shape[0], 1.5 * grid_shape[1]),
    )

    # Flatten the axes array if it's more than 1D
    if num_items > 1:
        axs = axs.flatten()

    # Iterate over the dictionary items and create the subplots
    for i, (title, data) in enumerate(dist_values.items()):
        ax = axs[i] if num_items > 1 else axs  # Use a single axis if there's only one item


        unique, counts = np.unique(data, return_counts=True)
        prob = counts/sum(counts)
        ax.scatter(unique, prob, color="#384265", s=30, alpha=0.3)

        # sns.kdeplot(data, ax=ax, fill=True, color="#384265")
        ax.set_title(format_title(title))
        
        ax.set_ylabel("Probability")

    # Remove any extra empty subplots, only if there is more than one distribution to plot
    if num_items > 1:
        if num_items < axs.size:
            for j in range(num_items, axs.size):
                fig.delaxes(axs[j])
            # Adjust spacing between subplots
    fig.tight_layout()
    fig.suptitle('Node-level Properties', y=1.02, fontsize=16)
    
    if verbose != None:
        set_log_level(dataviz_logger, current_level)

    return fig, axs


def plot_scalars(data_dict, verbose: str= None, title: str = None):
    num_ticks = 4
    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    # Extract keys (strings) and values (floats) from the dictionary
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    # Count the number of values below the threshold
    threshold = 0.05 * max(values)
    num_below_threshold = sum(value < threshold for value in values)
    # use log scale if more than 30% of the values are below the threshold
    use_log_scale = num_below_threshold > (len(values) / 3) and max(values) > 1
    



    dataviz_logger.info('Plotting global properties...')
    with sns.axes_style("darkgrid"):
        # Create the figure and axes
        fig, axs = plt.subplots(figsize=(2, 0.3 * len(labels)))

        if use_log_scale:
            plt.xscale("symlog", base=10)

        # Create a horizontal bar plot using seaborn
        sns.barplot(x=values, y=labels, ax=axs, color="#384265")

        # Add annotations to the bars
        for i, value in enumerate(values):
            if isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    value = round(value, 2) if value >= 0.01 else value
            value_srt = str(value) if value >= 0.01 or value==0 else f"{value:.2E}"
            plt.text(float(value), i, value_srt, va="center")

        # Customize the plot
        axs.set_xlim(0, max(values) * 1.1)
        axs.set_xlabel("Values")
        axs.set_ylabel("")

        if title is not None:
            axs.set_title("Network-level Properties", loc="center", fontsize=16)
        if use_log_scale:
            max_log = int(math.log10(ceil_to_next_power_of_10(max(values))))
            tick_locations = np.logspace(0, max_log, max_log-1)
        else:
            tick_locations = np.linspace(0, max(values), num_ticks)
        axs.set_xticks(tick_locations)
        # axs.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{int(value)}'))

    # plt.tight_layout()
    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    
    return fig, axs


# Plotting Fxns
def create_symmetric_heatmap(dataframe, title: str = None, method="ward", features=None, data_type=None, verbose: str = None):

    """Create a symmetric heatmap of the input dataframe.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe.
        title (str): The title of the heatmap.
        method (str): The method to use for clustering.
        features (pd.DataFrame): The input features dataframe. Index must be network names.
        data_type (dict): The data type of each feature.
        verbose (str): The verbosity level of the logger.

    Returns:
        fig (matplotlib.figure.Figure): The figure containing the heatmap.
        ax (matplotlib.axes._subplots.AxesSubplot): The axes containing the heatmap.
    
    """

    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    
    dataviz_logger.info('Creating symmetric heatmap...')
    # Create a figure and axes
    # fig, axs = plt.subplots()

    def add_features(features: pd.DataFrame, data_type: dict):
        color_mappings = {}
        norms = {}

        for col in features.columns:
            if data_type[col] == 'categorical':
                # Continue handling categorical columns as before
                unique_values = features[col].unique()
                mapping = dict(zip(unique_values, CATEGORICAL[:len(unique_values)]))
                color_mappings[col] = features[col].map(mapping)
            else:
                # Normalize and map each numerical column individually
                norm = Normalize(vmin=features[col].min(), vmax=features[col].max())
                norms[col] = norm  # Store the normalization
                cmap = sns.color_palette(NUMERICAL, as_cmap=True)
                mappable = ScalarMappable(norm=norm, cmap=cmap)
                color_mappings[col] = features[col].apply(lambda x: mappable.to_rgba(x))

        # Convert color mappings to DataFrame for row_colors
        row_colors = pd.DataFrame(color_mappings)

        # Generate clustermap
        g = sns.clustermap(dataframe, row_colors=row_colors, col_cluster=True, cmap="bone_r", yticklabels=False, xticklabels=False, figsize=(5, 5))

        # Add colorbars for numerical columns
        colorbar_width = 0.02 # each colorbar width 
        colorbar_spacing = 0.12  # Space between colorbars
        start_x_position = 1.02
        colorbar_counter = 0
        for col, norm in norms.items():
            cmap = sns.color_palette(NUMERICAL, as_cmap=True)
            mappable = ScalarMappable(norm=norm, cmap=cmap)
            current_x_position = start_x_position + colorbar_counter * (colorbar_width + colorbar_spacing)
            cbar_ax = g.figure.add_axes([current_x_position, .2, colorbar_width, .5])
            g.figure.colorbar(mappable, cax=cbar_ax, label=col)
            colorbar_counter += 1
            cbar_ax.set_ylabel(col, rotation=90, labelpad=2)

        # Add legends for categorical columns
        start_y_position = 1.0 
        # Aprox height of each legend entry
        legend_height = 0.1
        for index, (col, mapping) in enumerate(color_mappings.items()):
            if data_type[col] != 'categorical':
                continue
            unique_values = features[col].unique()
            handles = [mpl.patches.Patch(color=CATEGORICAL[i % len(CATEGORICAL)], label=val) for i, val in enumerate(unique_values)]
            current_y_position = start_y_position - (index * legend_height)
            g.figure.legend(handles=handles, title=col, loc='lower left', bbox_to_anchor=(1, current_y_position), ncol=int(len(unique_values)/2))

        return g

    if features is not None and data_type is None:
        dataviz_logger.critical('Data type of each feature is required to plot the heatmap with features. Please provide the data type of each feature as a dictionary.')

    elif features is not None and data_type is not None:
        try:
            g = add_features(features, data_type)
        except ValueError:
            dataviz_logger.critical('For one or more networks the properties array is constant. Correlation coefficient is not defined. Maybe adding more properties fro analysis...')
            raise ValueError('For one or more networks the properties array is constant. Correlation coefficient is not defined.')

    else:
        try: 
            g = sns.clustermap(
                dataframe.astype(float),
                cmap="bone_r",
                # vmin=0,
                vmax=1,
                annot=True if dataframe.shape[0] < 10 else False,
                fmt=".2f",
                cbar=True,
                method=method,
            )
        except ValueError:
            dataviz_logger.critical('For one or more networks the properties array is constant. Correlation coefficient is not defined. Maybe adding more properties fro analysis...')
            raise ValueError('For one or more networks the properties array is constant. Correlation coefficient is not defined.')

    # Set the title
    if title is not None:
        g.ax_heatmap.set_title(title, y=0, pad=-25, verticalalignment="top")

    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    # Return the figure and axes
    return g.figure #, g.ax_heatmap

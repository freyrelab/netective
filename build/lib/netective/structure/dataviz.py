from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from netective.logging_info import get_logger, set_log_level
from matplotlib.ticker import FuncFormatter
import logging

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
        nrows=grid_shape[0], ncols=grid_shape[1], figsize=(3 * grid_shape[0], 1.5 * grid_shape[1])
    )

    # Flatten the axes array if it's more than 1D
    if num_items > 1:
        axs = axs.flatten()

    # Iterate over the dictionary items and create the subplots
    for i, (title, data) in enumerate(dist_values.items()):
        ax = axs[i] if num_items > 1 else axs  # Use a single axis if there's only one item
        sns.kdeplot(data, ax=ax, fill=True, color="#384265")
        ax.set_title(format_title(title))

    # Remove any extra empty subplots, only if there is more than one distribution to plot
    if num_items > 1:
        if num_items < axs.size:
            for j in range(num_items, axs.size):
                fig.delaxes(axs[j])
            # Adjust spacing between subplots
            fig.tight_layout()
    
    if verbose != None:
        set_log_level(dataviz_logger, current_level)

    return fig, axs


def plot_scalars(data_dict, verbose: str= None):
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
    use_log_scale = num_below_threshold > (len(values) / 3)



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
        axs.set_title("Network Level Properties", loc="left")
        # axs.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{int(value)}'))

    plt.tight_layout()

    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    
    return fig, axs


# Plotting Fxns
def create_symmetric_heatmap(dataframe, title: str, method="ward", verbose: str = None):
    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    
    dataviz_logger.info('Creating symmetric heatmap...')
    # Create a figure and axes
    # fig, axs = plt.subplots()

    # Plot the heatmap
    try: 
        g = sns.clustermap(
            dataframe.astype(float),
            cmap="Blues",
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
    g.ax_heatmap.set_title(title)

    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    # Return the figure and axes
    return g.fig #, g.ax_heatmap

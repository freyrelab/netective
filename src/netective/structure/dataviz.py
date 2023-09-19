from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_distributions(dist_values):
    # Determine the grid shape based on the number of items
    num_items = len(dist_values)
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
            plt.text(value, i, str(value) if value >= 0.01 else f"{value:.2E}", va="center")

        # Customize the plot
        axs.set_xlabel("Values")
        axs.set_ylabel("")
        axs.set_title("Network Level Properties", loc="left")

    return fig, axs


# Plotting Fxns
def create_symmetric_heatmap(dataframe, title: str, method="ward"):
    # Create a figure and axes
    # fig, axs = plt.subplots()

    # Plot the heatmap
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

    # Set the title
    g.ax_heatmap.set_title(title)

    # Return the figure and axes
    return g.fig#, g.ax_heatmap

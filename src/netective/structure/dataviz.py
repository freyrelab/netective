from __future__ import annotations

from netective.logging_info import get_logger, set_log_level

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import seaborn as sns
import math
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

CATEGORICAL = sns.color_palette("Paired")
NUMERICAL = "rocket_r"

dataviz_logger = get_logger(__name__)

def ceil_to_next_power_of_10(number: int):
    return 10 ** math.ceil(math.log10(number))

def format_title(input_str: str):
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

# Plotting fxns
def plot_distributions(dist_values: dict, verbose: str = None, title: str = None):
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
    for i, (dist_title, data) in enumerate(dist_values.items()):
        ax = axs[i] if num_items > 1 else axs  # Use a single axis if there's only one item


        unique, counts = np.unique(data, return_counts=True)
        prob = counts/sum(counts)
        ax.scatter(unique, prob, color="#384265", s=30, alpha=0.3)

        # sns.kdeplot(data, ax=ax, fill=True, color="#384265")
        ax.set_title(format_title(dist_title))
        
        ax.set_ylabel("Probability")

    # Remove any extra empty subplots, only if there is more than one distribution to plot
    if num_items > 1:
        if num_items < axs.size:
            for j in range(num_items, axs.size):
                fig.delaxes(axs[j])
            # Adjust spacing between subplots
    fig.tight_layout()

    if title is not None:
        fig.suptitle('Node-level properties', y=1.02, fontsize=14)
    
    if verbose != None:
        set_log_level(dataviz_logger, current_level)

    return fig, axs

def plot_scalars(data_dict: dict, verbose: str= None, title: str = None):
    """_summary_

    _extended summary_[#_unique ID_]_

    .. math:: _LaTeX formula_

    Arguments:
        data_dict (dict): _description_
        verbose (str): _description_. Defaults to None.
        title (str): _description_. Defaults to None.

    Returns:
        _type_: _description_

    References:
        .. [#_unique ID_] _pubmed abbr journal title_ _vol_:_page or e-article id_ (_year_) https://doi.org/_doi_
        .. [#_unique ID_] _first-author first-name last-name_ _book title_ (_year_) ISBN:_ISBN_ _http link_
        .. [#_unique ID_] _article title_ _conference_ (_year_) _http link_"""
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
            axs.set_title('Network-level properties', loc="center", fontsize=16)
        if use_log_scale:
            max_log = int(math.log10(ceil_to_next_power_of_10(max(values))))
            tick_locations = np.logspace(0, max_log, max_log-1)
        else:
            tick_locations = np.linspace(0, max(values), num_ticks)
        axs.set_xticks(tick_locations)
        # axs.xaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{int(value)}'))

    plt.tight_layout()
    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    
    return fig, axs

def create_scalar_dist_plot(scalars_df: pd.DataFrame, scale: str = 'linear', add_box_plot: bool = False, title: str = None, verbose: str = None) -> matplotlib.figure.Figure:
    """Creates a plot of scalar properties, adapting its format based on the structure of the input DataFrame.

    This function generates a scatter or strip plot depending on the DataFrame type (either 'Groups' or 'Other').
    If `scalars_df` has a structure appropriate for plotting as returned by `netective.utils.get_scalar_props_df`,
    the plot will display scalar properties across different networks or groups. A box plot can be optionally added 
    to highlight the distribution of each property if DataFrame type = 'Other'.

    Arguments:
        scalars_df (pd.DataFrame): Dataframe containing scalar properties to be visualized.
        scale (str): Y-axis scale, either 'linear','log' or other (check matplotlib documentation
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale). Defaults to 'linear'.
        add_box_plot (bool): If True, adds a box plot overlay on top of the scatter plot for each property. Defaults to False.
        title (str): Title for the plot. If None, no title is displayed. Defaults to None.

    Raises:
        TypeError: If `scalars_df` does not contain the appropriate structure for plotting.

    Returns:
        matplotlib.figure.Figure: The generated plot figure, formatted according to the DataFrame type and input arguments."""

    # Returns Dataframe form for correct ploting
    def __verify_df_form(scalars_df:pd.DataFrame)-> Union[str, None]:
        """Verifies if DataFrame has the correct format and returns its shape.
        
        This function checks if the DataFrame has the correct format and returns its shape. Whether the DataFrame was 
        created by netective.utils.get_scalar_props_df or not, it must have the proper structure to be used in 
        netective.dataviz.create_scalar_props_plot .

        Shapes of Dataframes created by netective.utils.get_scalar_props_df :
            - Complete: A DataFrame containing all scalar property values for all networks, including (if part of scalars_array) 
                        the properties of random models. In this format, the index represents property names and columns 
                        represent Net IDs.
            - Filtered: Similar to the Complete format but excluding random models.
            - Groups:   A long-format DataFrame where the first column indicates the group to which each network belongs 
                        (either Input networks or a random model). The second column specifies the scalar property, and 
                        the third column stores the value of that property.
            - Averages: A DataFrame in which each column represents a specific group (either Input networks or a random model), 
                        and each cell contains the average value of all networks in that group for the corresponding property.
        
        Notes:
            - For plotting purposes, as Complete, Filtered, and Averages DataFrames are based on a similar structure, 
            this function only returns shape models as either 'Other' or 'Groups' if the DataFrame has the correct structure.
            - Even if the DataFrame was not created by netective.utils.get_scalar_props_df , it can still be plotted if it 
            follows the same structure as 'Other' or 'Groups'.
            - If the DataFrame does not have the appropriate structure, the function returns `None`.

        Arguments:
        scalars_df (pd.DataFrame): DataFrame to be used for plotting in `netective.dataviz.create_scalar_props_plot`.

        Returns:
            Union[str, None]: Returns 'Groups' or 'Other' depending on the structure of the DataFrame, 
                            or returns `None` if it does not follow an appropriate structure."""
    
        columns = list(scalars_df.columns)

        # Verify if DataFrame is group shaped (this is checked first because index == int)
        if columns == ['group', 'property', 'value']:
            return 'Groups'

        # Verify if DataFrame has the correct structure
        column_is_str = all(isinstance(col, str) for col in columns) 
        index_is_str = all(isinstance(idx, str) for idx in scalars_df.index)
        values_are_numerical = scalars_df.applymap(lambda x: isinstance(x, (int, float))).all().all()

        if not column_is_str or not index_is_str or not values_are_numerical:
            return None    
        
        return 'Other'

    # Set logger
    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    
    dataviz_logger.info('Creating comparison Scalar distribution plot...')

    # Verifies if scalars_df has the appropriate form for plotting
    shape_df = __verify_df_form(scalars_df)

    if shape_df == None:
        dataviz_logger.critical('scalars_df must have the appropriate shape or format so it can be ploted check netective.utils.get_scalar_props_df for details.')
        raise TypeError('scalars_df must have the appropriate shape or format so it can be ploted check netective.utils.get_scalar_props_df for details.')

    ### When type_dataframe introduced is not Groups

    if shape_df == 'Other':
        # Determine plot layout based on the number of elements
        long_label_size = 0 if len(max(list(scalars_df.columns), key=len)) < 40 else 2.5
        total_props = len(scalars_df.index)
        fig_width = round(total_props * 0.5)

        if total_props <= 6:
                gs = GridSpec(1,2,width_ratios= [2,3], height_ratios= [8])
                fig_width =  fig_width + 6.5 + long_label_size

        else:
                gs = GridSpec(1,2,width_ratios= [2,1], height_ratios= [8])
                fig_width =  fig_width + 3 + long_label_size
        fig = plt.figure(figsize=(fig_width, 10))
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])

        # Create scatter plot with property names as x labels
        sns.scatterplot(data=scalars_df, ax=ax1)
        ax1.set_xticks(ticks= range(len(scalars_df.index)),labels=scalars_df.index,rotation=90)
        ax1.set_ylabel(f"Values ({scale} scale)")

        # Optionally overlay box plot
        if add_box_plot:
            sns.boxplot(data=scalars_df.T, ax=ax1, fill=False , color= 'black', width=0.6, dodge=False, showfliers=False, linewidth=0.5, gap=.1) 
        else: 
            ax1.set_xlim(-0.5, len(scalars_df.index) - 0.5)
            
        # Display legend in separate axis if label
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend_.remove() 
        ax2.legend(handles, labels, loc='center left')
        ax2.axis('off')

        # Set scale 
        ax1.set_yscale(scale)

        # Add vertical lines between properties
        for x_pos in range(len(scalars_df.index)-1):
            ax1.axvline(x=x_pos + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha= 0.2)

        # Set title and layout
        ax1.set_title(title)
        plt.tight_layout()

        return fig
    
    ### When type_dataframe introduced is Groups

    # Calculate fig_width
    total_props = len(scalars_df['property'].unique())
    total_groups = len(scalars_df['group'].unique())

    fig_width = round(total_groups * total_props * 0.15) if total_props > 6 else round(total_groups * total_props * 0.15) + 3

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Create strip plot with grouping information
    sns.stripplot(data=scalars_df, x="property", y="value", hue="group", dodge=True, jitter=False, s=5, alpha=0.4, ax=ax)

    # Set axis labels and legend
    ax.set_xticks(range(len(ax.get_xticklabels())))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Property")
    ax.set_ylabel(f"Values ({scale} scale)")
    ax.set_title(title)
    ax.legend(title=None, framealpha=0)

    # Set scale
    ax.set_yscale(scale)

    # Add vertical lines between properties
    for x_pos in range(total_props - 1):
        ax.axvline(x=x_pos + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

    # Adjust legend location if there are fewer than 5 properties
    if total_props <= 6:
        # Move legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0) 
    else:
        # Default legend inside plot
        ax.legend(title=None, framealpha=0)
        
    # Adjust layout
    plt.tight_layout()
        
    if verbose != None:
        set_log_level(dataviz_logger, current_level)

    return fig


def create_comp_heatmap(distances_df: pd.DataFrame, title: str = None, metric: str= 'euclidean', method:str = "ward", features: pd.DataFrame= None, data_type: dict= None, verbose: str= None, compare_to_models: bool= None) -> matplotlib.figure.Figure:
    """Create a comparison heatmap of the input dataframe.

    Arguments:
        distances_df (pd.DataFrame): the input dataframe.
        title (str): title of the heatmap. Defaults to None.
        metric (str): distance metric to use . Defaults to 'euclidean'.
        method (str): method to use for clustering. Defaults to "ward".
        features (pd.DataFrame): input features dataframe. Defaults to None.
        data_type (dict): data type of each feature. Defaults to None.
        verbose (str): verbosity level of the logger. Defaults to None.
        compare_to_models (bool): whether the heatmap is comparing input networks to model analogs. Defaults to None.

    Raises:
        TypeError: if either features or data_type is provided and not the other.
        ValueError: when a properties array is constant. Not possible to calculate correlation.
        ValueError: when a properties array is constant. Not possible to calculate correlation.

    Returns:
        matplotlib.figure.Figure: figure containing the heatmap."""

    # Linkage
    row_linkage = linkage(distances_df, metric= metric, method= method)
    color_map = LinearSegmentedColormap.from_list('BlueWhiteRed', ['blue', 'white', 'red']) if compare_to_models else 'bone_r'

    if verbose != None:
        current_level = dataviz_logger.getEffectiveLevel()
        set_log_level(dataviz_logger, verbose)
    
    dataviz_logger.info('Creating comparison heatmap...')
    
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
        g = sns.clustermap(distances_df, row_linkage= row_linkage, col_linkage=row_linkage, row_colors= row_colors, col_cluster= False if compare_to_models else True, cmap= color_map, yticklabels= True, xticklabels= False if title else True, figsize= (8, 8), annot= True if distances_df.shape[0] < 10 else False, fmt= '.2f')
        
        # Add colorbars for numerical columns
        heatmap_bbox = g.ax_heatmap.get_position()
        dendrogram_bbox = g.ax_row_dendrogram.get_position()
        row_colors_bbox = g.ax_row_colors.get_position()
        colorbar_width = 0.02 # each colorbar width 
        colorbar_spacing = 0.12  # Space between colorbars
        start_x_position = dendrogram_bbox.width + row_colors_bbox.width + heatmap_bbox.width + 0.025
        colorbar_counter = 0
        for col, norm in norms.items():
            cmap = sns.color_palette(NUMERICAL, as_cmap=True)
            mappable = ScalarMappable(norm=norm, cmap= cmap)
            current_x_position = start_x_position + colorbar_counter * (colorbar_width + colorbar_spacing)
            cbar_ax = g.figure.add_axes([current_x_position, heatmap_bbox.y0, colorbar_width, heatmap_bbox.height])
            cbar = g.figure.colorbar(mappable, cax= cbar_ax, label= col)
            cbar.outline.set_visible(False)
            colorbar_counter += 1
            cbar_ax.set_ylabel(col, rotation=90, labelpad=2)

        # Add legends for categorical columns
        current_y_position = heatmap_bbox.height
        start_x_position = 0.1 * (colorbar_counter + 1) + 1
        legend_counter = 0
        legend_height = 0
        for col, mapping in color_mappings.items():
            if data_type[col] != 'categorical':
                continue
            unique_values = features[col].unique()
            handles = [Patch(color=CATEGORICAL[i % len(CATEGORICAL)], label=val) for i, val in enumerate(unique_values)]
            current_y_position -= legend_height
            current_x_position = start_x_position + (0.2 * legend_counter)
            # Aprox height of current legend entry
            legend_height = (len(unique_values) + 1) * 0.029
            g.figure.legend(handles=handles, title=col, ncol=1, loc= 'upper center', bbox_to_anchor= (start_x_position, current_y_position, 0.2, 0.05), mode= 'expand', frameon= False, alignment= 'left')
            legend_counter += 1
        
        return g

    if (features is not None and data_type is None) or (features is None and data_type is not None):
        dataviz_logger.critical('Both features DataFrame and data_type dictionary must be provided. Please provide a features DataFrame and the data type of each feature as a dictionary.')
        raise TypeError('Both features DataFrame and data_type dictionary must be provided. Please provide a features DataFrame and the data type of each feature as a dictionary.')

    elif features is not None and data_type is not None:
        try:
            g = add_features(features= features, data_type= data_type)
        except ValueError:
            dataviz_logger.critical('For one or more networks the properties array is constant. Correlation coefficient is not defined. Maybe adding more properties for analysis...')
            raise ValueError('For one or more networks the properties array is constant. Correlation coefficient is not defined.')

    else:
        try: 
            g = sns.clustermap(
                distances_df,
                row_linkage= row_linkage,
                col_linkage= row_linkage,
                cmap= color_map,
                # vmax= 1,
                annot= True if distances_df.shape[0] < 10 else False,
                fmt= '.2f',
                cbar= True,
                col_cluster= False if compare_to_models else True,
                yticklabels= True,
                xticklabels= False if title else True
            )
        except ValueError:
            dataviz_logger.critical('For one or more networks the properties array is constant. Correlation coefficient is not defined. Maybe adding more properties fro analysis...')
            raise ValueError('For one or more networks the properties array is constant. Correlation coefficient is not defined.')

    # Set the title
    if title:
        g.ax_heatmap.set_title(title, y=0, pad=-25, verticalalignment="top")

    if verbose != None:
        set_log_level(dataviz_logger, current_level)
    
    # Return the figure
    return g.figure
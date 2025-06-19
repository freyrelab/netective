from __future__ import annotations

import os
import warnings
import tracemalloc
import psutil
import gc
import inspect
import hashlib
import traceback
import numpy as np
import pandas as pd
import logging
import igraph as ig
import matplotlib.pyplot as plt
from itertools import chain
from networkx import Graph, DiGraph, number_of_nodes, number_of_edges, is_directed
from scipy.stats import pearsonr, rv_discrete
from collections import defaultdict
from multiprocessing import cpu_count
from typing import Callable, Tuple, Union
import uuid
from netective.structure import properties
from netective.utils import (
    compute_moments,
    run_parallel,
    validate_network,
    ShortestDistances,
    ShortestPaths,
    count_3motifs,
    giant_component,
    remove_self_loops,
    association,
    common_props_dict,
    get_clusters,
    sort_files,
    parse_network,
    get_allocated_memory,
    filter_association_df_for_models,
    get_models_abbreviations,
    clean_names_association_df,
    process_netective_properties_files
)
from netective.logging_info import get_logger
from netective.structure.dataviz import plot_scalars, create_comp_heatmap, plot_distributions

# Constants
NORM_OPTIONS = [None, "network", "biological"]
CORRELATION_OPTIONS = ['pearson', 'spearman', 'cosine']
PARENT_CLASS = properties._Property
MOTIFS = 16
DIRECTED = 8
SELF_LOOPS = 4
GIANT_COMPONENT = 2
PATHS = 1
MODELS = ["Erdos GNP", "Erdos GNM", "K Regular", "Barabasi Albert"]
BARABASI_M = ["out degree", "in degree", "degree"]

struct_logger = get_logger(__name__)

# Auxiliar Fxns
def set_log_level(verbose: str = 'WARNING'):
    """Logging level setter

    Function to change the verbosity level for the logging report.

    Arguments:
        verbose (str): The verbosity level of the logger. Defaults to 'WARNING'.
    """
    if isinstance(verbose, str):
        numeric_level = getattr(logging, verbose.upper(), None)
    else:
        numeric_level = verbose
    if not isinstance(numeric_level, int):
        struct_logger.critical(f'Invalid verbose level: {verbose}')
        raise TypeError(f'Invalid verbose level.')
    
    struct_logger.setLevel(numeric_level)

def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))

# Get properties selected Fxn
def get_child_classes(parent_class: type=properties._Property, selected_props: str|list="all", conserve_props: bool = True, include_env: None|dict=None) -> dict:
    """Returns a dict of child classes of parent_class based on selected_props.
    
    Arguments:
        parent_class (type): Parent class to search for child classes.
            This function is intented to work for the properties.Property abscract class.
        selected_props (str|list, optional): Properties to search for. Defaults to "all".
            Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names.
        conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
        include_env (None|dict, optional): Dictionary with the environment variables to include. Defaults to None.
    
    Returns:
        dict: Dictionary with the child classes found and their corresponding mask to process the graph.
    
    Raises:
        ValueError: If no valid property names are provided.
    """

    def _valid_child_cls(cls, parent_class):
        return inspect.isclass(cls) and issubclass(cls, parent_class) and cls != parent_class
    
    def _define_mask(cls):
        bool_mask = [
            cls._use_motifs,
            cls._use_direction,
            cls._use_selfloops,
            cls._use_giant_component,
            cls._use_paths
        ]
        return np.packbits(bool_mask).item() >> 3

    child_classes = {}
    all_properties = []
    property_childs = inspect.getmembers(properties)

    if include_env is not None:
        if not isinstance(include_env, dict):
            raise ValueError(f'Invalid environment variables. Must be a dictionary.')
        for key, value in include_env.items():
            if _valid_child_cls(value, parent_class):
                struct_logger.info(f'{key} is a user-defined property.')
                all_properties.append(key)
                child_classes[value] = _define_mask(value)
        
    struct_logger.info("Properties used for analysis (based on selected_props): ")

    for name, cls in property_childs:
        if _valid_child_cls(cls, parent_class):
            all_properties.append(cls.CLASS_NAME)
            child_classes[cls] = _define_mask(cls)

    if selected_props != "all":
        if not isinstance(selected_props, list):
            selected_props = [selected_props]
        invalid_props = [prop for prop in selected_props if prop not in all_properties]
        if invalid_props:
            struct_logger.warning(f"The following properties can't be computed: {invalid_props}, therefore will be eliminates from selected_props list.")
            selected_props=[prop for prop in selected_props if prop not in invalid_props]
        if not conserve_props:
            struct_logger.info(f"Removing following properties so they won't be computed: {selected_props}")
            selected_props = [prop for prop in all_properties if prop not in selected_props]
        child_classes = {cls: mask for cls, mask in child_classes.items() if cls.CLASS_NAME in selected_props}
    
    for cls in child_classes:
        struct_logger.info(f'{cls.CLASS_NAME}')
        
    struct_logger.info('')
    if len(child_classes) == 0:
        if conserve_props:
            struct_logger.critical(f"Sorry, no matches for properties inquired.\nThe available properties are: {all_properties}")
        else: 
            struct_logger.critical(f"List of selected properties (with parameter conserve_props = False) contain all available properties.\nTry again with less properies or with conserve_props parameter to deffault (True)")
        raise ValueError(
            f"Invalid list of properties."
        )
    return child_classes


class GraphObserver:

    """A class to observe changes in a graph."""

    def __init__(self, G: Graph | DiGraph, data: bool = False) -> None:
        """
        Initialize the GraphObserver class.

        Arguments:
            G (Graph | DiGraph): The graph to observe.
            data (bool): If True, the node/edge data will be considered when computing the hash.
        """
        self.G = G
        self.data = data
        self.graph_hash = self._compute_hash(self.G)

    def _compute_hash(self, G: Graph | DiGraph) -> str:
        """
        Compute the hash of the graph.

        Arguments:
            G (Graph | DiGraph): The graph to compute the hash.

        Returns:
            str: The hash of the graph.
        """

        hash = hashlib.sha1(
            f"nodes: {G.nodes(data=self.data)},\
            edges: {G.edges(data=self.data)}".encode(
                "utf-8"
            )
        )
        return hash.hexdigest()

    def changed(self, G: Graph | DiGraph = None, update_G: bool = False) -> bool:
        """
        Check if G has changed with reference to the last call.

        Arguments:
            G (Graph | DiGraph): The graph to check. If None, the original graph will be used.
            update_G (bool): If True, the graph will be updated to G. If False, the graph will not be updated.

        Returns:
            bool: True if the graph has changed, False otherwise.

        Raises:
            ValueError: If G is None and update_G is True.

        Note:
            If G is None and update_G is False, the original graph will be used (default behavior).

            The following table shows the behavior of the function:
            hash_org_G: hash of the original graph
            hash_new_G: hash of the new graph

            | G | update_G | return | update graph | update hash |
            |---|----------|--------|--------------|-------------|
            | None | False | hash_org_G == new_hash_org_G | False | True |
            | not None | True | hash_org_G == hash_new_G | True | True |
            | not None | False | hash_org_G == hash_new_G | False | False |
            | None | True | ValueError | False | False |
        """
        if update_G and G is None:
            raise ValueError("G cannot be None if update_G is True")

        tmp_G = self.G if G is None else G
        new_hash = self._compute_hash(tmp_G)

        if new_hash != self.graph_hash:
            change_flag = True

            if G is None:
                # update the hash of the original graph
                self.graph_hash = new_hash

            elif update_G:
                # G and update_G
                self.G = G
                self.graph_hash = new_hash

            # if not update_G and G is not None, no need to update the hash

        else:
            change_flag = False

            if update_G:
                # G and update_G
                self.G = G
                # no need to update the hash

        return change_flag


class NormObserver:
    """A class to observe changes in the normalization strategy."""

    def __init__(self, norm):
        """
        Initialize the NormObserver class.

        Arguments:
            norm (None | str | pd.Series): The norm to observe.
        """
        self.norm = norm
        self.norm_hash = self._compute_hash()

    def _hash(self, str_norm):
        """
        Compute the SHA-1 hash of a string.
        Implemented to DRY the code.

        Arguments:
            str_norm (str): The string to compute the hash.

        Returns:
            str: The hash of the string.
        """
        hash_object = hashlib.sha1(str_norm.encode("utf-8"))
        return hash_object.hexdigest()

    def _compute_hash(self):
        """
        Compute the SHA-1 hash of the current norm value

        Arguments:
            norm (None | str | pd.Series): The norm to compute the hash.

        Returns:
            str: The hash of the norm. If norm is None, return None.
        """
        if self.norm is None:
            return None

        elif isinstance(self.norm, str):
            str_norm = self.norm
            return self._hash(str_norm)

        elif isinstance(self.norm, pd.Series):
            # convert it to a flat string to be hashed
            # self.norm.to_string(index=True, dtype=True, name=True, length=True, header=True)
            str_norm = f"pd.Series: {self.norm}"
            return self._hash(str_norm)

    def change(self) -> bool:
        """
        Check if norm has changed with reference to the last call.

        Returns:
            bool: True if the norm has changed, False otherwise.
        """
        new_hash = self._compute_hash()
        if new_hash != self.norm_hash:
            change_flag = True
            self.norm_hash = new_hash
        else:
            change_flag = False
        return change_flag


class Structure:
    """
        Object to compute, normalize, and store the structural properties of a network.
        Structure inherits from pandas.Series.
    """
    def __init__(
        self,
        G: DiGraph | Graph,
        norm: None | str | pd.Series = None,
        net_id: str = None,
        verbose: str = None,
    ):
        """
        Creating a Structure object does not compute the structural properties.
        Instead, it sets the attributes of the object for future property computation.
        Use the get_props() method to compute the structural properties.
        If the network has changed, the properties and the normalization factors are recomputed when get_props() is called.

        Arguments:
            G (DiGraph | Graph): Network to compute the structural properties.
            norm (None | str | pd.Series): Normalization factor for each property. Defaults to None.
                Use None to disable normalization.
                Use 'biol' to normalize by the biological scale factors.
                Use a dictionary to normalize by custom scale factors.
                Missing properties are reported as NaN.
            net_id (str): Name of the network. If None, a random uuid is assigned.
                Used for verbose mode and raising errors.
            verbose (str, optional): The verbosity level of the logger. Defaults to None.

        Returns:
            Structure: Object with the structural properties of the network.

        Raises:
            TypeError: If G is not a DiGraph or a Graph.
            ValueError: If G has no edges.

        Examples:
            >>> from netective import Structure
            >>> S = Structure(nx.complete_graph(10), norm='biol')
            >>> S.get_props()   # returns a Series object with the structural properties normalized by the biological scale factors
        """
        self._original_G = G  # original network
        self.G = validate_network(
            self._original_G.copy()
        )  # network to compute the structural properties
        self.graph_observer = GraphObserver(
            G
        )  # note that it observes the original graph, not the copy
        self.graph_observer.graph_hash = None  # hash of the network to detect changes. None means that the properties have not been computed yet.
        self.norm_observer = NormObserver(
            norm
        )  # object to observe changes in the normalization strategy
        self.verbose = verbose
        self.net_id = net_id if net_id is not None else str(uuid.uuid4())[:8]

    @property
    def norm(self) -> None | str | pd.Series:
        """
        Normalization factor for each property.

        Returns:
            None|str|pd.Series: Normalization factor for each property.
        """
        return self.norm_observer.norm

    @norm.setter
    def norm(self, norm: None | str | pd.Series):
        """
        Modifies the normalization factor for each property.
        Future calls to get_props() will use the new normalization factor.

        Arguments:
            norm (None | str | pd.Series): Normalization factor for each property.
                                           Use None to disable normalization.
                                           Use 'biol' to normalize by the biological scale factors.
                                           Use a dictionary to normalize by custom scale factors.
                                           Missing properties are reported as NaN.
        """
        self.norm_observer.norm = norm

    def _normalize_props(self, instances, norm):
        """Normalizes the structural properties of a network."""

        struct_logger.info("Normalizing...")
        
        norm_scalar_values = {}
        norm_dist_values = {}
        if norm not in NORM_OPTIONS:
            struct_logger.critical(f"Invalid normalization method: {norm}")
            raise properties.NormalizationError(f"Invalid normalization method: {norm}")
        struct_logger.warning("Properties excluded from analysis due to lack of normalization:")
        for name, x in instances.items():
            dict_ = norm_scalar_values if x._return_type == "scalar" else norm_dist_values
            try:
                if norm == "network":
                    dict_[x.CLASS_NAME] = x.norm_network()
                elif norm == "biological":
                    dict_[x.CLASS_NAME] = x.norm_biol()
            except (NotImplementedError, properties.NormalizationError):
                # dict_[x.CLASS_NAME] = np.nan
                struct_logger.warning(f"{x.CLASS_NAME}")
                continue
        return norm_scalar_values, norm_dist_values

    def __get_modify_directed_graphs(self, property_groups, original_graph):
        graphs = {}

        for mask, class_group in property_groups.items():
            struct_logger.debug(f'Creating modified graph copy for {class_group} properties class group...')
            directed = mask & DIRECTED != 0

            # Modifications for undirected graphs in another fxn
            if not directed:
                continue

            haveto_remove_self_loops = mask & SELF_LOOPS == 0
            get_giant_component = mask & GIANT_COMPONENT != 0
            get_paths = mask & PATHS != 0
            get_motifs = mask & MOTIFS != 0

            # Dummy graph that will be modified, only if it applies
            graph_copy = original_graph.copy()

            # Graph modifications, only if it applies
            if haveto_remove_self_loops:
                graph_copy = remove_self_loops(graph_copy)
            if get_giant_component:
                graph_copy = giant_component(graph_copy)
            struct_logger.debug('Gettting aditional params...')
            if get_paths:
                struct_logger.debug('Creating <ShortestPaths> object...')
                net_shortest_paths = ShortestPaths(graph_copy)
                struct_logger.debug('Creating <ShortesDistances> object...')
                net_shortest_distances = ShortestDistances(graph_copy)
                # Input requires paths objects besides the modified graph
                graphs[mask] = (graph_copy, net_shortest_paths, net_shortest_distances)
            else:
                if get_motifs: # Input requires motifs object besides the modified graph
                    struct_logger.debug('Creating <3motifs> object...')
                    motifs_obj = count_3motifs(graph_copy)
                    graphs[mask] = (graph_copy, motifs_obj)
                else: # Input requires only the modified graph
                    graphs[mask] = graph_copy
        
        return graphs

    def __get_modify_undirected_graphs(self, property_groups, original_graph):
        graphs = {}
        for mask, class_group in property_groups.items():
            struct_logger.debug(f'Creating modified graph copy for {class_group} properties class group...')
            directed = mask & DIRECTED != 0

            # Modifications for directed graph in another fxn
            if directed:
                continue

            haveto_remove_self_loops = mask & SELF_LOOPS == 0
            get_giant_component = mask & GIANT_COMPONENT != 0
            get_paths = mask & PATHS != 0
            get_motifs  = mask & MOTIFS != 0

            # Dummy graph that will be modified, only if it applies
            graph_copy = original_graph.copy()

            # Graph modifications, only if it applies
            if haveto_remove_self_loops:
                graph_copy = remove_self_loops(graph_copy)
            if get_giant_component:
                graph_copy = giant_component(graph_copy)
            struct_logger.debug('Gettting aditional params...')
            if get_paths:
                struct_logger.debug('Creating <ShortestPaths> object...')
                net_shortest_paths = ShortestPaths(graph_copy)
                struct_logger.debug('Creating <ShortesDistances> object...')
                net_shortest_distances = ShortestDistances(graph_copy)
                # Input requires paths objects and motifs object besides the modified graph
                graphs[mask] = (graph_copy, net_shortest_paths, net_shortest_distances)
            else:
                if get_motifs: # Input requires motifs object besides the modified graph
                    struct_logger.debug('Creating <3motifs> object...')
                    motifs_obj = count_3motifs(graph_copy)
                    graphs[mask] = (graph_copy, motifs_obj)
                else: # Input requires only the modified graph
                    graphs[mask] = graph_copy
        
        return graphs

    def __get_instances(self, property_groups, original_graph):

        instances = {}
        modified_directed_graphs = {}
        modified_undirected_graphs = {}

        struct_logger.debug('Creating graph copies...')
        if original_graph.is_directed():
            struct_logger.debug('Directed graph detected. Creating modified copies...')
            modified_directed_graphs.update(
                self.__get_modify_directed_graphs(property_groups, original_graph)
            )
            struct_logger.debug('Removing direction. Creating modified copies...')
            modified_undirected_graphs.update(
                self.__get_modify_undirected_graphs(property_groups, original_graph.to_undirected())
            )
        else:
            struct_logger.debug('Undirected graph detected. Creating modified copies...')
            modified_undirected_graphs.update(
                self.__get_modify_undirected_graphs(property_groups, original_graph)
            )

        # Dict with {keys: masks for each property group}
        #           {values: required input to instance each property in that property group}
        inputs = {**modified_directed_graphs, **modified_undirected_graphs}

        for mask, class_group in property_groups.items():
            if mask not in inputs:
                for class_ in class_group:
                    struct_logger.warning(f"{class_.CLASS_NAME} cannot be computed for the input graph.")
                continue

            property_input = inputs[mask]

            for class_ in class_group:
                if isinstance(property_input, tuple):
                    G = property_input[0]
                    if isinstance(property_input[1], ShortestPaths):
                        net_shortest_paths = property_input[1]
                        net_shortest_distances = property_input[2]
                        instances[class_.CLASS_NAME] = class_(
                            G,
                            net_shortest_paths= net_shortest_paths,
                            net_shortest_distances= net_shortest_distances,
                        )
                    else:
                        motifs_obj = property_input[1]
                        instances[class_.CLASS_NAME] = class_(
                            G,
                            motifs_obj= motifs_obj
                        )
                else:
                    instances[class_.CLASS_NAME] = class_(property_input)
        return instances

    def _compute_props(self, child_classes) -> dict[str, float, int]:
        """Computes the structural properties of a network.

        Arguments:
            G (DiGraph | Graph): Network to compute the structural properties.
            norm (None | str | pd.Series): Normalization factor for each property.
            net_id (str): Name of the network. If None, a random uuid is assigned.
            child_classes (dict): Dict of classes to compute the structural properties.
                dict[str, float, int]

        Returns:
            dict: Dictionary with the structural properties of the network.
        """

        struct_logger.info(f"Processing {self.net_id}...")
        struct_logger.info(f"{self.net_id} has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

        if not self.G.is_directed() and self.norm_observer.norm == "biological":
            struct_logger.critical("Biological normalization is only available for directed graphs")
            raise properties.NormalizationError(
                "Biological normalization is only available for directed graphs"
            )

        property_groups = defaultdict(list)
        for class_, mask in child_classes.items():
            property_groups[mask].append(class_)

        instances = self.__get_instances(property_groups, self.G)

        self.scalar_values = {}
        self.dist_values = {}
        struct_logger.debug('Starting properties computation...')
        for prop_name, prop in instances.items():
            if prop._return_type == 'scalar':
                self.scalar_values[prop.CLASS_NAME] = prop.compute()
            else:
                self.dist_values[prop.CLASS_NAME] = prop.compute()
            struct_logger.debug(f'Finished computing: {prop.CLASS_NAME}')

        if self.norm_observer.norm is not None:
            self.scalar_values, self.dist_values = self._normalize_props(
                instances, norm=self.norm_observer.norm
            )
        
        # eliminate instances to free memory
        # TODO: Memory Optimization: Instaces are regenerated even when only the normalization has changed. Consider including a cache for the instances.
        del instances
        gc.collect()

        self.dist_values = {k: v for k, v in self.dist_values.items() if not np.isnan(v).all()}

        return self.scalar_values, self.dist_values

    def get_props(
        self, selected_props: str | list = "all", conserve_props: bool = True, child_classes: list = None, include_env: None | dict = None
    ) -> Tuple[dict[str, dict[str, float | int]], dict[str, dict[str, np.array]]]:
        """Get structural properties for the network instanced.

        Computes the structural properties of a network.
        Either props or child_classes must be provided.
        If both are provided, props is ignored.

        Arguments:
            selected_props (str|list): Structural properties to compute. Defaults to 'all'.
                Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names. Ignored if child_classes is provided.
            conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
            child_classes (dict): List of property classes to compute all the properties of those classes. If provided, selected_props and include_env is ignored. Defaults to None.
            include_env (None|dict, optional): Dictionary with the environment variables to include. Defaults to None. Ignored if child_classes is provided. Defaults to None.

        Returns:
            Tuple[dict[str, dict[str, float | int]], dict[str, dict[str, np.array]]]: Tuple of dictionaries with the network id and the properties, values of the network.
        """

        # Compute the properties if they have not been computed yet or if the network has changed
        if self.verbose != None:
            current_level = struct_logger.getEffectiveLevel()
            set_log_level(self.verbose)

        if (
            self.graph_observer.graph_hash is None
            or self.graph_observer.changed(self._original_G, update_G=True)
        ) or self.norm_observer.change():
            struct_logger.warning('The network or the normalization method has changed. Computing its properties...')
            # TODO Optimization: cache the raw values?
            # TODO: include a verbose message when normalization has changed

            # TODO: Refactor code: First run is None and self.graph_observer.changed(self._original_G is not evaluated due to bypass.
            if self.graph_observer.graph_hash is None:
                self.graph_observer.changed(self._original_G, update_G=True)

            self.G = validate_network(
                self._original_G.copy()
            )  # to make sure the network is valid and use the actual modified network
            try:
                self._scalar_arrays = {}
                self._dist_arrays = {}

                if child_classes is None:
                    child_classes = get_child_classes(PARENT_CLASS, selected_props, conserve_props, include_env=include_env)

                # props
                scalar_values, dist_values = self._compute_props(child_classes)
                self._scalar_arrays[self.net_id] = scalar_values
                self._dist_arrays[self.net_id] = dist_values

                return self._scalar_arrays, self._dist_arrays

            # This is a general exception handler to catch any error that may occur in the parallelized code
            except Exception as e:
                tracebackString = traceback.format_exc(e)
                struct_logger.critical(
                    f"Error occurred. Original traceback is\n{tracebackString}\n"
                )
                raise NotImplementedError(
                    f"\n\nError occurred. Original traceback is\n{tracebackString}\n"
                )
        
        if self.verbose != None:
            set_log_level(current_level)
        
        return self._scalar_arrays, self._dist_arrays

# Characterization of one network
def characterize_network(
    G: DiGraph | Graph,
    net_id: str = None,
    norm: str | None = None,
    selected_props: str | list = "all",
    conserve_props: bool = True,
    child_classes: dict = None,
    include_env: None | dict = None,
    return_prop_dicts: bool = False,
    verbose: str = None,
    title: str = None
) -> Tuple[plt.Figure, plt.Figure] | Tuple[dict[str, float | int], dict[str, np.array]]:
    """
    Module-level function to characterize a single network.

    Arguments:
        G (DiGraph | Graph): Network to characterize.
        norm (str, optional): Normalization to apply. Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
            Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names. Ignored if child_classes is provided.
        conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
        child_classes (dict, optional): Dict of child classes to compute. Defaults to None.
            if child_classes is not None, selected_props and include_env is ignored.
        include_env (None | dict, optional): Dictionary with the environment variables to include. Defaults to None. Ignored if child_classes is provided.
        verbose (str, optional): The verbosity level of the logger. Defaults to None.
            View logging levels from Logging library.
        return_prop_dicts (bool, optional): Whether to return the properties as dictionaries. Defaults to False.
            If False, the figures are shown.
        title (str): The title to include in both network-level and node-level properties figs. Defaults to None.

    Returns:
        Tuple: Tuple of figures with the properties of the network.
        Tuple: Tuple of dictionaries with the properties of the network if return_prop_dicts is True.

    Raises:
        Exception: Raised if the normalization is not valid.
    """
    if verbose != None:
        current_level = struct_logger.getEffectiveLevel()
        set_log_level(verbose)
    
    # TODO characterize_network should also be able to parse network from a file
    net_id = net_id if net_id is not None else str(uuid.uuid4())[:8]
    struc = Structure(G, norm= norm, net_id= net_id, verbose= verbose)
    if child_classes is not None:
        scalar_values, dist_values = struc.get_props(child_classes=child_classes)
    else:
        scalar_values, dist_values = struc.get_props(selected_props=selected_props, conserve_props=conserve_props, include_env=include_env)

    if len(dist_values) == 0 and len(scalar_values) == 0:
        struct_logger.critical("Not enough data, try with more properties or another normalization")
        raise ValueError("Not enough data, try with more properties or another normalization")

    if return_prop_dicts:
        return scalar_values, dist_values
    
    if len(dist_values) != 0:
        fig_dist, _ = plot_distributions(dist_values[net_id], verbose= verbose, title= title)

    if len(scalar_values) != 0:
        fig_scalar, _ = plot_scalars(scalar_values[net_id], verbose= verbose, title= title)
    
    if verbose != None:
        set_log_level(current_level)
    
    return fig_scalar, fig_dist

def __remove_network_data(G: Graph) -> Graph:
    """
    Remove all the data from a network.

    Arguments:
        G (Graph): Network to remove the data.

    Returns:
        Graph: Network without data.
    """
    G = G.copy()
    for n in G.nodes:
        G.nodes[n].clear()
    for u, v in G.edges:
        G.edges[u, v].clear()
    return G

def __get_optimal_workers(nets : str | dict, directed: bool, comments: str, delimiter: str, nets_file_format: str = 'edgelist') -> int:
    """Calculation of optimal workers for parallelization

    Computes a preliminary structural characterization and plotting of the biggest network from the inputed bunch of networks.
    Based on the allocated memory from this process, the total number of cpus and the total available memory at the time,
    an approximate optimal max number of workers is determined for the overall process. The possible number of workers is
    always determined from [1, total number of cpus). An aproximate 80% of total available memory is used as a maximal 
    threshold, not the entire available memory.

    .. math:: _LaTeX formula_

    Arguments:
        nets (str | dict): Dictionary of networks to compare | Path to directory containing multiple networks.
                                IMPORTANT: it is more memory efficient to pass networks as a path rather than a dictionary.
            {'net_id': DiGraph | Graph}
        directed (bool): Whether or not the network is directed.
        comments (str): Character used to indicate the start of a comment.
        delimiter (str): String used to separate values.

    Returns:
        int: Number of optimal workers for the parallelized process.
    """
    workers = cpu_count() - 1
    mem = psutil.virtual_memory()
    available_mem = mem.available / 1000000
    eighty_percent_available_mem = (available_mem * 50) / 100
    tracemalloc.start()
    if isinstance(nets, dict):
        max_edges = 0
        for net_id, net in nets.items():
            if net.number_of_edges() > max_edges:
                max_edges = net.number_of_edges()
                max_net = net_id
        net_id = max_net
        net = nets[net_id]
    
    elif os.path.isdir(nets):
        sorted_files = sort_files(path= nets)
        net_id = os.path.basename(sorted_files[0])
        net = parse_network(
            file_path= sorted_files[0],
            comments= comments,
            delimiter= delimiter,
            directed= directed,
            net_file_format= nets_file_format
        )
    foo, spam = characterize_network(
        G= net,
        net_id= net_id,
        verbose='critical',
        return_prop_dicts= True
    )
    for net_id, props in foo.items():
        fig_scalar, _ = plot_scalars(data_dict= props, verbose= 'critical')
    for net_id, props in spam.items():
        fig_dist, _ = plot_distributions(props, verbose= 'critical')
    snapshot = tracemalloc.take_snapshot()
    plt.close('all')
    mem_peak = get_allocated_memory(snapshot, filtered= False)
    for i in range(workers, 0, -1):
        if (mem_peak * i) < eighty_percent_available_mem:
            return i
    return 1

def __batch_processing(
        networks: dict,
        norm: str,
        selected_props: list,
        conserve_props: bool,
        child_classes: list,
        verbose: str,
        workers: int,
        keep_averages: bool,
        include_env: dict
    ) -> Tuple[dict[str, float | int], dict[str, np.array]]:
    """Batch processing for parallelization of the structural characterization process

    _extended summary_[#_unique ID_]_

    .. math:: _LaTeX formula_

    Arguments:
        networks (dict): _description_
        norm (str): _description_
        selected_props (list): _description_
        conserve_props (bool): _description_
        child_classes (list): _description_
        verbose (str): _description_
        workers (int): _description_
        keep_averages (bool): _description_
        include_env (dict): _description_
        
    Returns:
        Tuple[dict[str, float | int], dict[str, np.array]]: _description_
    """
    networks = {net_id: __remove_network_data(G) for net_id, G in networks.items()} # to avoid serialization error py3.8 with nx's data structures
    data = [
        list(networks.values()),
        list(networks.keys()),
        [norm] * len(networks),
        [selected_props] * len(networks),
        [conserve_props] * len(networks),
        [child_classes] * len(networks),
        [include_env] * len(networks),
        [True] * len(networks),
        [verbose] * len(networks),
    ]

    # run parallel
    results = run_parallel(characterize_network, data, workers, verbose= verbose)
    name_scalars_array = results["scalars"]
    name_dist_arrays = results["distributions"]

    name_moments_arrays = {
        net_id: { prop_name: compute_moments(array) for prop_name, array in prop.items() }
        for net_id, prop in name_dist_arrays.items()
    }

    if keep_averages:
        for net_id, prop in name_moments_arrays.items():
            for prop_name, values in prop.items():
                name_scalars_array[net_id][f'Average {prop_name}'] = values[0]
                # name_scalars_array[net_id][f'Variation {prop_name}'] = values[1]
                # name_scalars_array[net_id][f'Skewness {prop_name}'] = values[2]
                # name_scalars_array[net_id][f'Kurtosis {prop_name}'] = values[3]
    
    return name_scalars_array, name_dist_arrays

# Random Graph Analog Generator
def avg_random_nets_per_net(
    G: DiGraph | Graph | str,
    net_id: str,
    norm: None | str | pd.Series,
    random_model: str | function = None,
    number_of_random_nets: int =  2,
    directed_models: bool = False,
    random_graph_parameters: dict = None,
    ba_m: str | int = 2,
    selected_props : str | list = 'all',
    conserve_props: bool = True,
    keep_averages: bool = True,
    workers: int = 2,
    verbose: str = None,
    include_env: None | dict = None,
    nets_file_format: str = 'edgelist',
    comments: str = '#',
    delimiter: str = '\t',
    directed: bool = True,
    ) -> Tuple[dict[str, float | int], dict[str, np.array]]:
    """Module-level function to generate and characterize an x number of graph analog networks based on a random model generator.

    Generates an x number of networks based on a an specific model generator using as base some of the characteristics of a given graph.

    .. math:: $$ k = \frac{(2 * n_edges)}{n_nodes} $$ extracted from: $$ edges = \frac{(nodos * k)}{2} $$ [1]

    Arguments:
        G (DiGraph | Graph | str): network (directed or not) to create random analogs from. | Path to file containing edge list.
        net_id (str): Name given to the network introduced.
        norm (None | str | pd.Series): Normalization to apply. Defaults to None.
            Valid values are 'network', 'biological' or None.
        random_model (str | function): The random graph generator model that is going to be used.
            If model required not available, user can introduce their own function generator with the condition it retuns an nx.Graph() or an ig.Graph(). 
            Choices are "Erdos GNP", "Erdos GNM", "K Regular", "Barabasi Albert". Defaults to None.
        number_of_random_nets (int): Number of random graphs to generate with some of G properties. Defaults to 2.
        directed_models (bool): Whether analog models will be created using direction, if possible. Dafaults to False.
        random_graph_parameters (dict): If random_model is a fuction given by user, user must also give the parameter in a dictionary. Defaults to None.
        ba_m (str | int): Type of m (integrer or a degree distribution) that is going to be used in Barabasi Albert generator. Defaults to 2.
            Choices are "out degree", "in degree", "degree" or any positive integrer.
        selected_props (str | list):  Properties to compute. Defaults to 'all'.
        conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
        keep_averages (bool): Whether to keep the averages of the moments of the distributions. Defaults to True.
        workers (int): Number of workers to use. Defaults to 2.
                                IMPORTANT : Introducing a number of workers bigger than available is not going to make 
                                            the function break, but it will make batch creation incorrect, so computing 
                                            random models properties may be a little bit slower and consume more memory
                                            than necesary.
        verbose (str): The verbosity level of the logger.
            View logging levels from Logging library.
        include_env (None | dict): Dictionary with the environment variables to include. Defaults to None.
        nets_file_format (str): format for the networks files to parse.
        comments (str): Comment character in edge list file. Defaults to '#'.
        delimiter (str): Delimiter character in edge list file. Defaults to '\t'.
        directed (bool): If True, the network will be a DiGraph, otherwise it will be a Graph. Defaults to True.
    
    Notes:
        For "K Regular" algorithm the parameter k has to fulfill the condition: (k * n) % 2 == 0, where k is the final degree for each node and n is the total number of nodes [2]. 
        Therefore if condition is not met, k will be incremented by a unit.
        Seed condition was not implemented because igraph's random models generators doesn't include this parameter.

    Returns:
        Tuple[dict[str, float | int], dict[str, np.array]]: tuple of dictionaries(average scalar properties, average distributions moments)

    References:
        ..[1] Albert R., & Barabási A. L. Statistical mechanics of complex networks.(2002). Reviews of modern physics, Vol. 74(1), p.67. 
        ..[2] Svante J., Tomasz L., & Andrzej R. Random Graphs. (2000). JOHN WILEY & SONS, INC. p.3.
    """
    if verbose != None:
        current_level = struct_logger.getEffectiveLevel()
        set_log_level(verbose)
    
    if callable(random_model):
        struct_logger.info(f"Starting creation and characterization of {number_of_random_nets} random nets analogs to {net_id} with {random_model.__name__} function...")
    else:
        struct_logger.info(f"Starting creation and characterization of {number_of_random_nets} random nets analogs to {net_id} with {random_model} model...")

    # Function to convert from an iGraph object to a networkX graph.
    def conversion_ig_to_nx(n_nodes, ig_graph, directed):
        nx_graph = DiGraph() if directed else Graph()
        nx_graph.add_nodes_from(range(n_nodes))
        nx_graph.add_edges_from(ig_graph.get_edgelist())

        return nx_graph

    # Fuction to generate a list of edges from a given degree distribution and using them to create a Barabasi Albert graph
    def barabasi_dist_generator(n_nodes, pk_dist, directed):
        edges = [pk_dist.rvs() for _ in range(n_nodes)]
        ig_graph = ig.GraphBase.Barabasi(n = n_nodes, m = edges, directed = directed)

        return ig_graph
    
    # Veryfing if model generator exists
    if random_model not in MODELS and not callable(random_model):
        struct_logger.warning(f"Model: {random_model} not recognized as an available model generator.")
        return None
    
    # Veryfing if the G introduced is valid:
    if not isinstance(G, (Graph, DiGraph)):

        # If G is a file_path veryfing if path exists.
        if isinstance(G,str):
            try:
                G = parse_network(file_path= G, comments= comments, delimiter= delimiter, directed= directed, net_file_format= nets_file_format)
            except:
                struct_logger.warning(f"Error computing graph in path {G}, please verify if file path.")
                return None 
        
        else:
            struct_logger.warning(f"G must be a networkX graph or digraph, or a file path, imposible to create random_networks.")
            return None 
    
    # Veryfing if it is possible to generate an especific bunch of random networks.
    if number_of_random_nets <= 0:
        struct_logger.warning(f"Number of random nets generated must be a positive integrer bigger than 0, changing it to default (2).")
        number_of_random_nets = 2
    
    # Veryfing that number of workers is bigger than 0
    if workers <= 0:
        struct_logger.warning(f"Number of workers must be a positive integrer bigger than 0, changing number of workers to default (2).")
        workers = 2
    
    n_nodes = number_of_nodes(G)
    n_edges = number_of_edges(G)

    if callable(random_model):
        struct_logger.debug(f"Veryfing that {net_id} analogs can be created by {random_model.__name__} function.")
    else:
        struct_logger.debug(f"Veryfing that {net_id} analogs can be created by {random_model} model.")
    
    if random_model == 'Erdos GNP': # May include direction

        # Veryfing if n_nodes are distict of 0 so it can be posible to calculate density of network (p used in model).
        if n_nodes == 0:
            struct_logger.warning(f"{net_id} cannot be created using {random_model}. Cannot determine a p from an empty graph.")
            return None
        
        # Veryfing if graph is directed or not to calculate correctly the probability
        if is_directed(G):
            p = n_edges / (n_nodes**2)
        else:
            p =  n_edges/ (n_nodes ** 2 // 2) 

        model_generator = ig.GraphBase.Erdos_Renyi
        random_graph_parameters = {
                'n' : n_nodes,
                'p' : p,
                'directed' : directed_models
        }
        model_name = "GNP"

    elif random_model == 'Erdos GNM': # May include direction
        max_edges = (n_nodes * (n_nodes - 1)) // 2

        # Veryfing that the number of edges introduced to the model are less than n * (n-1) // 2 (max edges in an undirected, no self-loops graph).
        if n_edges > max_edges:
            struct_logger.warning(f"{net_id} has more edges than the posible to be computed by {random_model} model. Creating network with {max_edges} edges.")
            n_edges = max_edges
        
        model_generator = ig.GraphBase.Erdos_Renyi
        random_graph_parameters = {
                'n' : n_nodes,
                'm' : n_edges,
                'directed' : directed_models
        }
        model_name = "GNM"

    elif random_model == 'K Regular': # May be directed
        
        # Veryfing if n_nodes are distict of 0 so it can be posible to calculate average degree.
        if n_nodes == 0:
            struct_logger.warning(f"{net_id} cannot be created using {random_model}. Cannot determine a k from an empty graph.")
            return None
        
        avg_degree = round((2 * n_edges) / n_nodes)

        # Veryfing if the average degree corresponds to a complete graph with self-loops (k = n), if it does then k is going to be computed as n-1.
        if avg_degree >= n_nodes:
            struct_logger.warning(f"{net_id} used with {random_model} generator changed the value of k from {avg_degree} to {n_nodes - 1}. k mustn't be equal to n.")
            avg_degree = n_nodes - 1
         
        if not (avg_degree * n_nodes) % 2 == 0:
            # The average degree is increased by a single unit so the algorithm is able to resolve network creation.
            struct_logger.warning(f"{net_id} used with {random_model} generator changed k from {avg_degree} to {avg_degree + 1} so k * n equals an even number.")
            avg_degree += 1
        
        model_generator = ig.GraphBase.K_Regular
        random_graph_parameters = {
                'n' : n_nodes,
                'k' : avg_degree,
                'directed' : directed_models
        }
        model_name = 'KR'

    elif random_model == 'Barabasi Albert': # May include direction

        if isinstance(ba_m, int):

            # Veryfing that the m introduced is a positive integrer
            if ba_m < 0:
                struct_logger.warning(f"The m ({ba_m}) introduced for the Barabasi Albert is invalid, please introduce a positive integrer.")
                return None
            
            model_generator = ig.GraphBase.Barabasi
            random_graph_parameters = {
                    'n' : n_nodes,
                    'm' : ba_m,
                    'directed' : directed_models
            }
            model_name = f'BA-{ba_m}'
        
        else:

            # Veryfing that the distribution introduced is valid
            if ba_m not in BARABASI_M:
                struct_logger.warning(f"The m ({ba_m}) introduced for the Barabasi Albert is invalid.")
                return None
            
            # Veryfing if it possible to obtain a degree distribution from the graph.
            if n_nodes == 0:
                struct_logger.warning(f"{net_id} cannot be created by {random_model}-distribution model. Cannot determine degree distribution from an empty graph.")
                return None
            
            # Veryfing if it possible to obtain a directed degree distribution from the graph.
            if (ba_m == 'out degree' or ba_m == 'in degree') and not is_directed(G):
                struct_logger.warning(f"Not posible to get out or in degree distribution, because {net_id} is undirected.")
                return None
            
            model_generator = barabasi_dist_generator
            if ba_m == "out degree":
                degrees = np.array([degree for _, degree in G.out_degree()])
                model_name = "BA-out"
            elif ba_m == "in degree":
                degrees = np.array([degree for _, degree in G.in_degree()])
                model_name = "BA-in"
            elif ba_m == "degree":
                degrees = np.array([degree for _, degree in G.degree()])
                model_name = "BA-degree"
            xk, counts = np.unique(degrees, return_counts= True)
            pk = counts * (1 / G.number_of_nodes())
            pk_dist = rv_discrete(name= 'pk_dist', values= (xk, pk))
            random_graph_parameters = {
                    'n_nodes' : n_nodes,
                    'pk_dist' : pk_dist,
                    'directed' : directed_models
            }

    else:
        # Verification to check if function model is valid
        try:
            G_test = random_model(**random_graph_parameters)
        except:
            struct_logger.warning(f"The function {random_model.__name__} or random_graph_parameters are incorrect or not valid.")
            return None
        if not isinstance(G_test, (Graph, DiGraph)) and not isinstance(G_test,ig.GraphBase):
            struct_logger.warning(f"The function {random_model.__name__} is not valid, please return a networkX graph or digraph, or an igraph.")
            return None
        
        model_generator = random_model
        model_name = random_model.__name__
    
    name_scalars_array = {}
    name_distributions_arrays = {}
    
    # Counting total number of batches necesary to use minimum space in memory (create only nets that are going to be characterized).
    complete_batches = number_of_random_nets // workers
    last_batch = number_of_random_nets % workers
    
    if last_batch:
        total_batches = complete_batches + 1
        struct_logger.debug(f"Dividing the total number of nets = {number_of_random_nets} into {complete_batches} batches of {workers} nets + one last batch of {last_batch} nets usign {model_name} model, created from: {net_id}...")
    else:
        total_batches = complete_batches
        struct_logger.debug(f"Dividing the total number of nets = {number_of_random_nets} into {total_batches} batches of {workers} nets usign {model_name} model, created from: {net_id}...")
    
    struct_logger.warning("--------------------------------------------------------------------------------")
    struct_logger.warning(f"Starting creation and characterization of {number_of_random_nets} networks usign {model_name} model, created from: {net_id}...")

    # Creating the exact number of nets per batch and sending them to characterize:
    for batch in range(total_batches):
        inicial_net = batch * workers
        if batch < complete_batches:
            final_net = inicial_net + workers
        else:
            final_net = inicial_net + last_batch
        
        random_networks = {}

        struct_logger.debug(f"Creating nets of batch no.{batch + 1} and characterizing them...")

        # Creating nets necesary per batch using the introduced model
        for i in range(inicial_net,final_net):
            temp_g = model_generator(**random_graph_parameters)
            if isinstance(temp_g,ig.GraphBase):
                temp_g = conversion_ig_to_nx(n_nodes, temp_g, directed_models)
            random_networks[f'{i}_{model_name}_{net_id}'] = temp_g

        # Sending batch to characterize 
        temp_name_scalars_array, temp_name_distributions_arrays = compare_structure(
                            networks= random_networks,
                            norm= norm,
                            keep_averages= keep_averages,
                            selected_props= selected_props,
                            conserve_props=conserve_props,
                            workers= workers,
                            return_prop_dicts= True,
                            verbose= verbose,
                            include_env= include_env,
                            directed= directed_models
        )

        # Saving the characteristics of every net in every batch in 2 dictionaries one for the scalar properties and the other one for distributions
        name_scalars_array.update(temp_name_scalars_array)
        name_distributions_arrays.update(temp_name_distributions_arrays)
    
    del random_networks

    scalars_avg_random_net = {}
    moments_avg_random_net = {}

    struct_logger.debug(f"Calculating  the average value of every scalar property ...")

    # Determining averages for all scalar properties computed for each random network
    properties = {}
    for i,(temp_net_id, prop) in enumerate(name_scalars_array.items()):
        for prop_name, value in prop.items():
            if i == 0:
                properties[prop_name] = []
            properties[prop_name].append(value)
    scalars_props_avg = {
        prop_name : sum(values) / len(values)
        for prop_name, values in properties.items()
    }

    del name_scalars_array

    struct_logger.debug(f"Calculating the moments of each computed distribution...")

    # Calculating distributions moments for each random network
    name_moments_arrays = {}
    for temp_net_id, prop  in name_distributions_arrays.items():
            name_moments_arrays[temp_net_id] = {}
            for prop_name, array in prop.items():
                    name_moments_arrays[temp_net_id][prop_name] = compute_moments(array)
    
    del name_distributions_arrays
    
    struct_logger.debug(f"Calculating the average value of every distribution moment...")

    # Generating the lists of every computed moment for each property
    properties = {}
    for i, (temp_net_id, prop)  in enumerate(name_moments_arrays.items()):
        for prop_name, values in prop.items():
            if i == 0:
                properties[prop_name] = {
                    'Average' : [],
                    'Variation' : [],
                    'Skewness' : [],
                    'Kurtosis' : []
                }
            properties[prop_name]['Average'].append(values[0])
            properties[prop_name]['Variation'].append(values[1])
            properties[prop_name]['Skewness'].append(values[2])
            properties[prop_name]['Kurtosis'].append(values[3])
    
    del name_moments_arrays

    # Calculating the average of every computed moment for each property
    moments_props_avg = {}
    for prop_name, moments in properties.items():
        moments_props_avg[prop_name] = []
        for moment, values in moments.items():
            moment_avg = sum(values) / len(values)
            moments_props_avg[prop_name].append(moment_avg)
    
    # Saving the average of all scalar properties and distributions moments in a dictionaries with the format: {Avg_model_net : {property : value} }
    scalars_avg_random_net.update({f'Avg_{model_name}_{net_id}': scalars_props_avg})
    moments_avg_random_net.update({f'Avg_{model_name}_{net_id}': moments_props_avg})

    if verbose != None:
        set_log_level(current_level)

    return scalars_avg_random_net, moments_avg_random_net              

def characterize_models(
    networks: dict | str,
    norm: None | str | pd.Series,
    include_models: str | function | list = None,
    n_random_models : int = 2,
    directed_models: bool = False,
    random_graph_generator_params : dict = None,
    ba_m : int | str | list = 2,
    selected_props : str | list = 'all',
    conserve_props: bool = True,
    keep_averages: bool = True,
    workers: int = 2,
    verbose: str = None,
    include_env: None | dict = None,
    nets_file_format: str = 'edgelist',
    comments: str = '#',
    delimiter: str = '\t',
    directed: bool = True,
    ) -> Tuple[dict[str, float | int], dict[str, np.array]]:
    """This functions calls avg_random_nets_per_net with distinct models and parameters.

    Calls avg_random_nets_per_net with every model and conditions introduced for every net in a directory or dictionary of networks, and joins the result of every call in one dictionary.

    Arguments:
        networks (dict | str): _description_
        norm (None | str | pd.Series): Normalization to apply. Defaults to None.
            Valid values are 'network', 'biological' or None.
        include_models (str | function | list): List of or random graph generator model that is going to be used. Defaults to None.
        n_random_models (int): Number of random graphs to generate per model and per net. Defaults to 2.
        directed_models (bool): Whether analog models will be created using direction, if possible. Dafaults to False.
        random_graph_generator_params (dict): Dictionary of parameter for random model generator if random_model is a fuction given by user. Defaults to None.
        ba_m (int | str | list): Type of m (integrer or a degree distribution) that is going to be used in Barabasi Albert generator. Defaults to 2.
            Valid values are "out degree", "in degree", "degree" or any positive integrer.
        selected_props (str | list):  Properties to compute. Defaults to 'all'.
        conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
        keep_averages (bool): Whether to include the averages of local properties in the global properties array. Defaults to True.
        workers (int): Number of workers to use. Defaults to 2.
                                IMPORTANT : Introducing a number of workers bigger than available is not going to make 
                                            the function break, but it will make batch creation incorrect, so computing 
                                            random models properties may be a little bit slower and consume more memory
                                            than necesary.
        verbose (str): The verbosity level of the logger.
            View logging levels from Logging library.
        include_env (None | dict): Dictionary with the environment variables to include. Defaults to None.
        nets_file_format (str): format for the networks files to parse.
        comments (str): Comment character in edge list file. Defaults to '#'.
        delimiter (str): Delimiter character in edge list file. Defaults to '\t'.
        directed (bool): If True, the network will be a DiGraph, otherwise it will be a Graph. Defaults to True.

    Returns:
        Tuple[dict[str, float | int], dict[str, np.array]]: tuple of dictionaries(average scalar properties, average distributions moments)
                                                            containing the average value of scalar properties and distributions moments for
                                                            every model and condition of every network introduced."""

    if isinstance(networks,str):
        sorted_files = sort_files(networks)
        networks = {os.path.basename(net_path): net_path for net_path in sorted_files}

    if not isinstance(include_models, list):
        include_models = [include_models]

    if not isinstance(ba_m, list):
        ba_m = [ba_m]

    avg_nets_scalars_arrays = {}
    avg_nets_moments_arrays = {}

    for net_id, net in networks.items():
        for model in include_models:
            if model == "Barabasi Albert":
                for m in ba_m:
                    temp_dicts =  avg_random_nets_per_net(
                                                G= net,
                                                net_id = net_id,
                                                norm = norm,
                                                random_model = model,
                                                number_of_random_nets =  n_random_models,
                                                directed_models= directed_models,
                                                random_graph_parameters = random_graph_generator_params,
                                                ba_m = m,
                                                selected_props = selected_props,
                                                conserve_props=conserve_props,
                                                keep_averages = keep_averages,
                                                workers=workers,
                                                verbose= verbose,
                                                include_env= include_env,
                                                nets_file_format= nets_file_format,
                                                comments = comments,
                                                delimiter = delimiter,
                                                directed = directed,
                                                )
                    if temp_dicts is not None:
                        avg_nets_scalars_arrays.update(temp_dicts[0])
                        avg_nets_moments_arrays.update(temp_dicts[1])
            else:
                temp_dicts =  avg_random_nets_per_net(
                                            G= net,
                                            net_id = net_id,
                                            norm = norm,
                                            random_model = model,
                                            number_of_random_nets =  n_random_models,
                                            directed_models= directed_models,
                                            random_graph_parameters = random_graph_generator_params,
                                            selected_props = selected_props,
                                            conserve_props=conserve_props,
                                            keep_averages = keep_averages,
                                            workers=workers,
                                            verbose= verbose,
                                            include_env= include_env,
                                            nets_file_format= nets_file_format,
                                            comments = comments,
                                            delimiter = delimiter,
                                            directed = directed,
                                            )
                if temp_dicts is not None:
                        avg_nets_scalars_arrays.update(temp_dicts[0])
                        avg_nets_moments_arrays.update(temp_dicts[1])

    return avg_nets_scalars_arrays, avg_nets_moments_arrays

# Comparison of multiple networks
def compare_structure(
    networks: dict | str,
    directed: bool = True,
    norm: str | None = None,
    selected_props: str | list = "all",
    conserve_props: bool = True,
    workers: str | int = "auto",
    include_env: None | dict = None,
    return_prop_dicts: bool = False,
    return_association_df: bool = False,
    keep_averages: bool = True,
    association_metric: str | Callable = 'pearson',
    metric: str = 'euclidean',
    method: str = 'ward',
    include_models: str | list = None,
    compare_to_models: bool = False,
    n_random_models : int = 2,
    directed_models : bool = False,
    random_graph_generator_params : dict = None,
    ba_m : int | str | list = 2,
    verbose: str = None,
    nets_file_format: str = 'edgelist',
    comments : str = '#',
    delimiter : str = '\t',
    features: pd.DataFrame = None,
    data_type: dict = None,
    title: str = None,
    **clustermap_kwargs
) -> Union[Tuple[dict, dict], Tuple[dict, dict, dict, dict], Tuple[plt.Figure, pd.DataFrame]]:
    """Structural characterization-based networks' comparison

    _extended summary_[#_unique ID_]_

    .. math:: _LaTeX formula_

    Arguments:
        networks (dict | str): _description_
        directed (bool): _description_. Defaults to True.
        norm (str | None): _description_. Defaults to None.
        selected_props (str | list): _description_. Defaults to "all".
        conserve_props (bool): _description_. Defaults to True.
        workers (str | int): _description_. Defaults to "auto".
        include_env (None | dict): _description_. Defaults to None.
        return_prop_dicts (bool): _description_. Defaults to False.
        return_association_df (bool): _description_. Defaults to False.
        keep_averages (bool): _description_. Defaults to True.
        association_metric (str | Callable): _description_. Defaults to 'pearson'.
        metric (str): _description_. Defaults to 'euclidean'.
        method (str): _description_. Defaults to 'ward'.
        include_models (str | list): _description_. Defaults to None.
        compare_to_models (bool): _description_. Defaults to False.
        n_random_models (int): _description_. Defaults to 2.
        directed_models (bool): _description_. Defaults to False.
        random_graph_generator_params (dict): _description_. Defaults to None.
        ba_m (int | str | list): _description_. Defaults to 2.
        verbose (str): _description_. Defaults to None.
        nets_file_format (str): _description_. Defaults to 'edgelist'.
        comments (str): _description_. Defaults to '#'.
        delimiter (str): _description_. Defaults to '\t'.
        features (pd.DataFrame): _description_. Defaults to None.
        data_type (dict): _description_. Defaults to None.
        title (str): _description_. Defaults to None.

    Raises:
        properties.NormalizationError: _description_
        ValueError: _description_

    Returns:
        Union[Tuple[dict, dict], Tuple[dict, dict, dict, dict], Tuple[plt.Figure, pd.DataFrame]]: _description_

    References:
        .. [#_unique ID_] *_pubmed abbr journal title_* _vol_:_page or e-article id_ (_year_) https://doi.org/_doi_
        .. [#_unique ID_] _first-author first-name last-name_ *_book title_* (_year_) ISBN:_ISBN_ _http link_
        .. [#_unique ID_] _article title_ _conference_ (_year_) _http link_"""
    
    if isinstance(networks, dict) and len(networks) == 0:
        struct_logger.critical("Networks dictionary is empty.")
        raise ValueError('Networks dictionary is empty.')

    if verbose != None:
        current_level = struct_logger.getEffectiveLevel()
        set_log_level(verbose)
    
    if norm not in NORM_OPTIONS:
        struct_logger.critical("Normalization not valid")
        raise properties.NormalizationError("Normalization not valid")
    
    if association_metric not in CORRELATION_OPTIONS:
        struct_logger.warning(f'Correlation metric: {association_metric}, not valid. Setting default pearson correlation coefficient.')
        association_metric = 'pearson'

    # currently, both selected_props and child_classes are being passed to get_props, however, only one is needed.
    # TODO: Optimization:  passing only child_classes would be more efficient beacuse it computes get_child_classes only once.
    child_classes = get_child_classes(PARENT_CLASS, selected_props, conserve_props, include_env=include_env)

    # handle workers
    usable_workers = cpu_count() - 1
    if workers == "auto" or workers > usable_workers:
        struct_logger.warning('Getting optimal number of workers based on available memory and inputed networks sizes...')
        workers = __get_optimal_workers(
            nets= networks,
            directed= directed,
            nets_file_format= nets_file_format,
            comments= comments,
            delimiter= delimiter
        )

    # Processing of input networks
    name_scalars_array = {}
    name_dist_arrays = {}
    # networks is a directory path, transforming into dict, NOT parsing networks yet
    if isinstance(networks, str):
        sorted_files = sort_files(networks)
        networks = {os.path.basename(net_path): net_path for net_path in sorted_files}
    
    # Calculations for batch generation
    net_ids = list(networks.keys())
    complete_batches = len(net_ids) // workers
    last_batch = len(net_ids) % workers
    if last_batch:
        total_batches = complete_batches + 1
    else:
        total_batches = complete_batches
    
    # Batch processing of input networks
    for batch_number in range(total_batches):
        temp_nets = {}
        inicial_net = batch_number * workers
        if batch_number < complete_batches:
            final_net = inicial_net + workers
        else:
            final_net = inicial_net + last_batch
        
        struct_logger.debug(f"Creating nets of batch no.{batch_number + 1}/{total_batches} and characterizing them...")
        # Batch generation
        for i in range(inicial_net, final_net):
            net_id = net_ids[i]
            if isinstance(networks[net_id], str):
                temp_nets[net_id] = parse_network(
                    file_path= networks[net_id],
                    net_file_format= nets_file_format,
                    comments= comments,
                    delimiter= delimiter,
                    directed= directed
                )
            else:
                temp_nets[net_id] = networks[net_id]
        
        # Topological caracterization of batch 
        struct_logger.warning(f'Starting topological characterization of networks: {list(temp_nets.keys())}...')
        temp_arrays = __batch_processing(
                networks= temp_nets,
                norm= norm,
                selected_props= selected_props,
                conserve_props= conserve_props,
                child_classes= child_classes,
                verbose= verbose,
                workers= workers,
                keep_averages= keep_averages,
                include_env= include_env,
            )
        name_scalars_array.update(temp_arrays[0])
        name_dist_arrays.update(temp_arrays[1])
        del temp_nets

    # In case there are directed and undirected networks in input bunch
    name_scalars_array = common_props_dict(name_scalars_array)

    # Analog models generation 
    if include_models is not None:
        avg_nets_scalars_arrays, avg_nets_moments_arrays = characterize_models(
            networks= networks,
            norm= norm,
            include_models= include_models,
            n_random_models= n_random_models,
            directed_models= directed_models,
            random_graph_generator_params=random_graph_generator_params,
            ba_m=ba_m,
            selected_props=selected_props,
            conserve_props=conserve_props,
            keep_averages=keep_averages,
            workers=workers,
            verbose=verbose,
            include_env=include_env,
            nets_file_format= nets_file_format,
            comments=comments,
            delimiter=delimiter,
            directed=directed
        )
        avg_nets_scalars_arrays = common_props_dict(avg_nets_scalars_arrays)
    
    # Returning of props dict
    if return_prop_dicts:
        if verbose != None:
                set_log_level(current_level)
        if include_models:
            return name_scalars_array, name_dist_arrays, avg_nets_scalars_arrays, avg_nets_moments_arrays
        else:
            return name_scalars_array, name_dist_arrays
        
    if len(name_scalars_array) > 0 and len(list(name_scalars_array.values())[0]) > 1:
        if include_models:
            name_scalars_array.update(avg_nets_scalars_arrays)
            name_scalars_array = common_props_dict(name_scalars_array)
        
        association_df = association(name_scalars_array, corr_func= association_metric)
        
        if compare_to_models: # DF filtering required
            abbreviations = get_models_abbreviations(avg_nets_scalars_arrays)
            association_df = filter_association_df_for_models(association_df, abbreviations)
        
        association_df = clean_names_association_df(association_df)
        if features is not None and data_type is not None:
            features = clean_names_association_df(features)
            data_type = pd.DataFrame(data_type, index=[0])
            data_type = clean_names_association_df(data_type)
            data_type = data_type.to_dict(orient='records')[0]
            
        if return_association_df:
            return association_df
        fig_scalar = create_comp_heatmap(association_df, metric= metric, method= method, title= title, features= features, data_type= data_type, verbose= verbose, compare_to_models= compare_to_models, **clustermap_kwargs)
    else:
        struct_logger.critical("Not enough data to compare.")
        raise ValueError("Not enough data to compare.")
    
    if verbose != None:
        set_log_level(current_level)
    
    return fig_scalar, association_df

######################################################################
def classify_networks(
        networks: dict[str, Graph] | str = None,
        results_dir: str = None,
        distance_df: pd.DataFrame = None,
        norm: str | None = None,
        directed: bool = True,
        selected_props: str | list = 'all',
        conserve_props: bool = True,
        workers: str | int = "auto",
        include_env: None | dict = None,
        add_averages: bool = True,
        association_metric: str = 'pearson',
        clust_num: int = None,
        threshold: float = 0.7,
        metric: str = 'euclidean',
        method: str = 'ward',
        map_ids: bool = True,
        fcluster_kwargs: dict = None,
        verbose: str = None,
        nets_file_format: str = 'edgelist',
        comments : str = '#',
        delimiter : str = '\t',
) -> dict:
    """Module-level function to classify multiple networks.
    
    Returns groups of networks with similar properties.

    Arguments:
        networks (dict | str): Dictionary of networks to compare or directory path to files.
            {'net_id': DiGraph | Graph}
        norm (str, optional): Normalization to apply. Defaults to None.
            Valid values are 'network', 'biological' or None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
        conserve_props (bool): Whether to include or remove the selected properties. Defaults to True.
        workers (int, optional): Number of workers to use. Defaults to 'auto'.
            Auto means number of cpu's - 1.
        get_clusters_kargs (dict, optional): Keyword arguments for the get_clusters function. Defaults to None.

    Raises:
        NormalizationError: Raised if the normalization is not valid.
        ValueError: Raised if there is not enough data to compare.
    
    Returns:
        dict: Dictionary with the id of the cluster and the networks that belong to it.
    """
    if verbose != None:
        current_level = struct_logger.getEffectiveLevel()
        set_log_level(verbose)
    
    if distance_df is None:
        distance_df = pd.DataFrame()
    
    if not networks and not results_dir and distance_df.empty:
        struct_logger.critical('Either a directory/dictionary with networks, or a results directory with properties files created by Netective, or a distance dataframe created by Netective must be passed to classify networks.')
        raise ValueError('Either a directory/dictionary with networks, or a results directory with properties files created by Netective, or a distance dataframe created by Netective must be passed to classify networks.')
    
    if networks:
        struct_logger.warning('INPUT DETECTED: networks directory. Therefore entire structural characterization needs to be computed first.')
    elif results_dir:
        struct_logger.warning("INPUT DETECTED: Netective's results directory. Therefore only files processing and association need to be computed.")
    else:
        struct_logger.warning('INPUT DETECTED: distance dataframe. Therefore classification will be computed directly.')
    
    # Calculating properties arrays if necesary
    if not distance_df.empty: # In case a distance_df is already passed, no need to do anything else
        pass
    elif results_dir: # Scenario where structural characterization has already been computed
        scalars_array = process_netective_properties_files(
            results_dir= results_dir,
            return_props_dict= True,
            selected_props= selected_props,
            conserve_props= conserve_props,
            add_averages= add_averages,
            verbose= verbose
        )
    elif networks: # Worst case scenario, entire structural characterization needs to be computed
        scalars_array, _ = compare_structure(
            networks= networks,
            directed= directed,
            norm= norm,
            selected_props= selected_props,
            conserve_props= conserve_props,
            workers= workers,
            include_env= include_env,
            return_prop_dicts= True,
            keep_averages= add_averages,
            verbose= verbose,
            nets_file_format= nets_file_format,
            comments= comments,
            delimiter= delimiter
        )
    else: # Error, no input found
        struct_logger.critical('No input found. Please provide either a pre-computed distance dataframe, directory path with Netective pre-computed properties files or directory path with network files to run classification.')
        raise AttributeError('No input found. Please provide either a pre-computed distance dataframe, directory path with Netective pre-computed properties files or directory path with network files to run classification.')
    
    # Calculate distance dataframe if necessary
    if distance_df.empty:
        if len(scalars_array) > 0 and len(list(scalars_array.values())[0]) > 1:
            distance_df = association(scalars_array, corr_func= association_metric)
            distance_df = clean_names_association_df(distance_df)
        else:
            struct_logger.critical("Not enough data to compare.")
            raise ValueError("Not enough data to compare.")
        

    clusters = get_clusters(
        distance_df= distance_df,
        clust_num= clust_num,
        threshold= threshold,
        metric= metric,
        method= method,
        map_ids= map_ids,
        fcluster_kwargs= fcluster_kwargs
    )
    
    if verbose != None:
        set_log_level(current_level)

    return clusters
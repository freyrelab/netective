from __future__ import annotations

import time
import os
import gc
import uuid
import inspect
import hashlib
import traceback
import numpy as np
import pandas as pd
from typing import Tuple
from warnings import warn
from networkx import Graph
from itertools import chain
from networkx import DiGraph
from scipy.stats import pearsonr
from collections import defaultdict
from multiprocessing import cpu_count
from networkx import connected_components
from networkx import fast_gnp_random_graph

from netective.structure import properties
from netective.utils import (
    compute_moments,
    run_parallel,
    validate_network,
    concat_path,
    ShortestDistances,
    ShortestPaths,
    count_3motifs,
    giant_component,
    remove_self_loops,
    association,
)

import logging

from netective.logging_info import get_logger

from netective.structure.dataviz import plot_scalars, create_symmetric_heatmap, plot_distributions

import matplotlib.pyplot as plt

from typing import Callable

# Constants
NORM_OPTIONS = [None, "network", "biological"]
PARENT_CLASS = properties._Property
MOTIFS = 16
DIRECTED = 8
SELF_LOOPS = 4
GIANT_COMPONENT = 2
PATHS = 1

struct_logger = get_logger(__name__)

# Auxiliar Fxns
def set_log_level(verbose: str = 'CRITICAL'):
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
def get_child_classes(parent_class: type=properties._Property, selected_props: str|list="all", include_env: None|dict=None) -> dict:
    """Returns a dict of child classes of parent_class based on selected_props.
    
    Args:
        parent_class (type): Parent class to search for child classes.
            This function is intented to work for the properties.Property abscract class.
        selected_props (str|list, optional): Properties to search for. Defaults to "all".
            Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names.
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
        child_classes = {cls: mask for cls, mask in child_classes.items() if cls.CLASS_NAME in selected_props}
    
    for cls in child_classes:
        struct_logger.info(f'{cls.CLASS_NAME}')
        
    struct_logger.info('')
    if len(child_classes) == 0:
        struct_logger.critical(f"Sorry, no matches for properties inquired.\nThe available properties are: {all_properties}")
        raise ValueError(
            f"Invalid list of properties."
        )
    return child_classes


class GraphObserver:

    """A class to observe changes in a graph."""

    def __init__(self, G: Graph | DiGraph, data: bool = False) -> None:
        """
        Initialize the GraphObserver class.

        Args:
            G (nx.Graph | nx.Graph): The graph to observe.
            data (bool): If True, the node/edge data will be considered when computing the hash.

        """
        self.G = G
        self.data = data
        self.graph_hash = self._compute_hash(self.G)

    def _compute_hash(self, G: Graph | DiGraph) -> str:
        """
        Compute the hash of the graph.

        Args:
            G (nx.Graph | nx.DiGraph): The graph to compute the hash.
            data (bool): If True, the node/edge data will be considered when computing the hash.

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

        Args:
            G (nx.Graph | nx.DiGraph): The graph to check. If None, the original graph will be used.
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

        Args:
            norm (None | str | pd.Series): The norm to observe.

        """
        self.norm = norm
        self.norm_hash = self._compute_hash()

    def _hash(self, str_norm):
        """
        Compute the SHA-1 hash of a string.
        Implemented to DRY the code.

        Args:
            str_norm (str): The string to compute the hash.

        Returns:
            str: The hash of the string.
        """
        hash_object = hashlib.sha1(str_norm.encode("utf-8"))
        return hash_object.hexdigest()

    def _compute_hash(self):
        """
        Compute the SHA-1 hash of the current norm value

        Args:
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

    def change(self):
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
    def __init__(
        self,
        G: DiGraph | Graph,
        norm: None | str | pd.Series = None,
        net_id: str = None,
        verbose: str = None,
    ):

        """
        Object to compute, normalize, and store the structural properties of a network.
        Structure inherits from pandas.Series.

        Creating a Structure object does not compute the structural properties.
        Instead, it sets the attributes of the object for future property computation.
        Use the get_props() method to compute the structural properties.
        If the network has changed, the properties and the normalization factors are recomputed when get_props() is called.

        Args:
            G (DiGraph | Graph): Network to compute the structural properties.
            norm (None | str | pd.Series): Normalization factor for each property. Defaults to None.
                Use None to disable normalization.
                Use 'biol' to normalize by the biological scale factors.
                Use a dictionary to normalize by custom scale factors.
                Missing properties are reported as NaN.
            net_id (str): Name of the network. If None, a random uuid is assigned.
                Used for verbose mode and raising errors.
            verbose (str, optional): Whether to print information about the network. Defaults to None.

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

        # super().__init__()  # DataFrame.__init__(self)
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

        Args:
            norm: None|str|pd.Series.
                Normalization factor for each property.
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
        temp_times = []

        for mask, class_group in property_groups.items():
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
            if get_paths:
                net_shortest_paths = ShortestPaths(graph_copy)
                net_shortest_distances = ShortestDistances(graph_copy)
                # Input requires paths objects besides the modified graph
                graphs[mask] = (graph_copy, net_shortest_paths, net_shortest_distances)
            else:
                if get_motifs: # Input requires motifs object besides the modified graph
                    inicio = time.time()
                    motifs_obj = count_3motifs(graph_copy)
                    temp_times.append(time.time() - inicio)
                    graphs[mask] = (graph_copy, motifs_obj)
                else: # Input requires only the modified graph
                    graphs[mask] = graph_copy
        
        return graphs, temp_times

    def __get_modify_undirected_graphs(self, property_groups, original_graph):
        graphs = {}
        temp_times = []
        for mask, class_group in property_groups.items():
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
            if get_paths:
                inicio = time.time()
                net_shortest_paths = ShortestPaths(graph_copy)
                temp_times.append(time.time() - inicio)
                inicio = time.time()
                net_shortest_distances = ShortestDistances(graph_copy)
                temp_times.append(time.time() - inicio)
                # Input requires paths objects and motifs object besides the modified graph
                graphs[mask] = (graph_copy, net_shortest_paths, net_shortest_distances)
            else:
                if get_motifs: # Input requires motifs object besides the modified graph
                    motifs_obj = count_3motifs(graph_copy)
                    graphs[mask] = (graph_copy, motifs_obj)
                else: # Input requires only the modified graph
                    graphs[mask] = graph_copy
        
        return graphs, temp_times

    def __get_instances(self, property_groups, original_graph):
        instances = {}
        modified_directed_graphs = {}
        modified_undirected_graphs = {}

        if original_graph.is_directed():
            temp_dic, directed_times = self.__get_modify_directed_graphs(property_groups, original_graph)
            modified_directed_graphs.update(temp_dic)
            temp_dic, undirected_times = self.__get_modify_undirected_graphs(property_groups, original_graph.to_undirected())
            modified_undirected_graphs.update(temp_dic)
        else:
            modified_undirected_graphs.update(
                self.__get_modify_undirected_graphs(property_groups, original_graph)
            )
        unified_times = directed_times + undirected_times

        # Dict with keys: masks for each property group
        #           values: required input to instance each property in that property group
        inputs = {**modified_directed_graphs, **modified_undirected_graphs}

        for mask, class_group in property_groups.items():
            if mask not in inputs:
                for class_ in class_group: # Será sólo un warning
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
        return instances, unified_times

    def _compute_props(self, child_classes) -> dict[str, float, int]:
        """
        Computes the structural properties of a network.

        Args:
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

        instances, obj_creation_times = self.__get_instances(property_groups, self.G)
        process_times = {}
        process_times['motifs obj'] = obj_creation_times[0]
        process_times['shortest paths obj'] = obj_creation_times[1]
        process_times['shortest distances'] = obj_creation_times[2]

        self.scalar_values = {}
        self.dist_values = {}
        struct_logger.debug('Starting properties computation...')
        for name,x in instances.items():
            inicio = time.time()
            if x._return_type == 'scalar':
                self.scalar_values[x.CLASS_NAME] = x.compute()
            else:
                self.dist_values[x.CLASS_NAME] = x.compute()
            process_times[x.CLASS_NAME] = time.time() - inicio
            struct_logger.debug(f'Finished computing: {x.CLASS_NAME}')

        if self.norm_observer.norm is not None:
            self.scalar_values, self.dist_values = self._normalize_props(
                instances, norm=self.norm_observer.norm
            )
        
        # eliminate instances to free memory
        # TODO: Memory Optimization: Instaces are regenerated even when only the normalization has changed. Consider including a cache for the instances.
        del instances
        gc.collect()

        self.dist_values = {k: v for k, v in self.dist_values.items() if not np.isnan(v).all()}

        return self.scalar_values, self.dist_values, process_times

    def get_props(
        self, selected_props: str | list = "all", child_classes: list = None, include_env: None | dict = None
    ) -> Tuple[dict[str, dict], dict[str, dict]]:
        """
        Computes the structural properties of a network.
        Either props or child_classes must be provided.
        If both are provided, props is ignored.

        Args:
            selected_props (str|list): Structural properties to compute.
                Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names. Ignored if child_classes is provided.
            child_classes (dict): List of property classes to compute all the properties of those classes. If provided, selected_props and include_env is ignored.
            include_env (None|dict, optional): Dictionary with the environment variables to include. Defaults to None. Ignored if child_classes is provided.

        Returns:
            Tuple[dict, dict]: Tuple of dictionaries with the network id and the properties, values of the network.
        """
        # Compute the properties if they have not been computed yet or if the network has changed
        # print(
        #         f", is None?: {self.graph_observer.graph_hash is None}, graph changed: {self.graph_observer.changed(self._original_G, update_G=True)}, norm changed: {self.norm_observer.change()}"
        #     )
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
                self._dist_moments_arrays = {}

                if child_classes is None:
                    child_classes = get_child_classes(PARENT_CLASS, selected_props, include_env=include_env)

                # props
                scalar_values, dist_values, process_times = self._compute_props(child_classes)
                self._scalar_arrays[self.net_id] = scalar_values
                self._dist_moments_arrays[self.net_id] = {
                    prop_name: compute_moments(array) for prop_name, array in dist_values.items()
                }

                return self._scalar_arrays, self._dist_moments_arrays, process_times

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
        
        return self._scalar_arrays, self._dist_moments_arrays, process_times

def er_nets_per_net_analysis(
    G: DiGraph | Graph,
    net_id: str,
    norm: None | str | pd.Series,
    erdos_renyi: int = 2,
    selected_props : str | list = 'all',
    workers: str | int = "auto",
    verbose: str = None,
    include_env: None | dict = None,
) -> Tuple[dict, dict]:

    """
    Call the function er_nets_per_net_analysis to generate erdos_renyi number of ER networks for a given network.
    The function will compute properties for all ER networks generated, then calculate averages for each property.

    It returns a tuple of dictionaries, one for the average scalar properties and one for the average moments of each distribution.

    Args:
        G (DiGraph or Graph): Network to use as temple for ER networks created.
        norm (str, optional): Normalization to apply.
            Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
        erdos_renyi (int): Number of random graphs to generate with the same number of nodes and density as G. Defaults to 2.
        workers (int, optional): Number of workers to use. Defaults to 'auto'.
            Auto means number of cpu's - 1. 
        verbose (str, optional): Level of verbose desired for logging process.
            View logging levels from Logging library.

    Returns:
        Tuple[dict, dict]: Tuple of dictionaries with the network id and the properties of the network.
    """

    # Creating erdos_renyi number of ER networks, with the same number of nodes, density and direction
    n = G.number_of_nodes()
    m = G.number_of_edges()
    er_networks = {
        f'{net_id}_{i}' : fast_gnp_random_graph(n, m / (n**2), directed= G.is_directed())
        for i in range(erdos_renyi)
    }
    
    # Computing properties for erdos_renyi number of ER networks created
    name_er_scalars_array, name_er_moments_arrays = compare_structure(
                            networks= er_networks,
                            norm= norm,
                            selected_props= selected_props,
                            workers= workers,
                            return_prop_dicts= True,
                            verbose= verbose,
                            erdos_renyi= None,
                            include_env= include_env
    )

    # Determining averages for all scalar properties computed for each ER network
    properties = {}
    for i,(temp_net_id, prop) in enumerate(name_er_scalars_array.items()):
        for prop_name, value in prop.items():
            if i == 0:
                properties[prop_name] = []
            properties[prop_name].append(value)
    scalars_props_avg = {
        prop_name : sum(values) / erdos_renyi
        for prop_name, values in properties.items()
    }
    
    # Determining averages for all distribution properties computed for each ER network
    properties = {}
    for i, (temp_net_id, prop)  in enumerate(name_er_moments_arrays.items()):
        for prop_name, values in prop.items():
            if i == 0:
                properties[prop_name] = {
                    'Average' : [],
                    'Variation' : [],
                    'Skewness' : [],
                    'Kurtosis' : []
                }
            for k, value in enumerate(values):
                if k == 0:
                    moment = 'Average'
                elif k == 1:
                    moment = 'Variation'
                elif k == 2:
                    moment = 'Skewness'
                elif k == 3:
                    moment = 'Kurtosis'
                properties[prop_name][moment].append(value)

    dist_props_avg = {}
    for prop_name, moments in properties.items():
        dist_props_avg[prop_name] = []
        for moment, values in moments.items():
            moment_avg = sum(values) / len(values)
            dist_props_avg[prop_name].append(moment_avg)
    
    # Final dictionaries
    scalars_avg_er_net = {
        f'{net_id}_Avg_ER' : scalars_props_avg
    }
    dist_avg_er_net = {
        f'{net_id}_Avg_ER' : dist_props_avg
    }

    return scalars_avg_er_net, dist_avg_er_net


def save_strucs(
    scalar_props: dict,
    dist_props: dict,
    output_dir: str = os.getcwd(),
    delimiter: str = "\t",
    cl: str = None,
) -> None:
    """
    Save the structural properties in a file containing the name of the network and the properties.

    Args:
        scalar_props (dict): Dictionary with the scalar properties of the networks.
            {network_name: {property_name: property_value}}.
        dist_props (dict): Dictionary with the distribution properties of the networks.
            {network_name: {property_name: property_moments}}.
        output_dir (str): Path to the output directory. Defaults to current directory.
        delimiter (str): Delimiter to use in the output file. Defaults to tab.
        cl (str): Command line used to run the script.

    Returns:
        None.
    """

    exts = {",": "csv", "\t": "tsv"}
    ext = exts.get(delimiter, "txt")
    # file_p = concat_path(output, f"{output_file}.{ext}")

    network_name = list(scalar_props.keys())[0]

    file_p_s = concat_path(output_dir, f"{network_name}_scalar_props.{ext}")
    file_p_d = concat_path(output_dir, f"{network_name}_dist_props.{ext}")

    if cl is not None:
        with open(file_p_s, "w") as f:
            f.write(f"# {cl}\n")
        with open(file_p_d, "w") as f:
            f.write(f"# {cl}\n")

    # save scalar props as csv
    df_s = pd.DataFrame.from_dict(scalar_props, orient="index")
    df_s.to_csv(file_p_s, sep=delimiter)

    # save dist props as csv
    df_d = pd.DataFrame.from_dict(dist_props, orient="index")
    df_d.to_csv(file_p_d, sep=delimiter)
    

# User Fxn's
# Characterization of one network
def characterize_network(
    G: DiGraph | Graph,
    net_id: str = None,
    norm: str | None = None,
    selected_props: str | list = "all",
    child_classes: dict = None,
    include_env: None | dict = None,
    return_prop_dicts: bool = False,
    verbose: str = None,
) -> None | Tuple[dict, dict]:
    """
    Module-level function to characterize a single network.

    Args:
        G (DiGraph | Graph): Network to characterize.
        norm (str, optional): Normalization to apply. Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
            Use 'all' to search for all child classes. Otherwise, use the name of a property or a list of property names. Ignored if child_classes is provided.
        child_classes (dict, optional): Dict of child classes to compute. Defaults to None.
            if child_classes is not None, selected_props and include_env is ignored.
        include_env (None | dict, optional): Dictionary with the environment variables to include. Defaults to None. Ignored if child_classes is provided.
        verbose (str, optional): Level of verbose desired for logging process. Defaults to None.
            View logging levels from Logging library.
        return_prop_dicts (bool, optional): Whether to return the properties as dictionaries. Defaults to False.
            If False, the figures are shown.

    Returns:
        dict: Dictionary with the properties of the network if return_prop_dicts is True.
        tuple: Tuple of figures with the properties of the network if return_prop_dicts is False.

    Raises:
        Exception: Raised if the normalization is not valid.
    """
    if verbose != None:
        current_level = struct_logger.getEffectiveLevel()
        set_log_level(verbose)
    
    net_id = net_id if net_id is not None else str(uuid.uuid4())[:8]
    struc = Structure(G, norm= norm, net_id= net_id, verbose= verbose)
    if child_classes is not None:
        scalar_values, dist_values = struc.get_props(child_classes=child_classes)
    else:
        scalar_values, dist_values = struc.get_props(selected_props=selected_props, include_env=include_env)

    if len(dist_values) == 0 and len(scalar_values) == 0:
        struct_logger.critical("Not enough data, try with more properties or another normalization")
        raise ValueError("Not enough data, try with more properties or another normalization")

    if return_prop_dicts:
        return scalar_values, dist_values
    
    if len(dist_values) != 0:
        fig_dist, _ = plot_distributions(dist_values[net_id])

    if len(scalar_values) != 0:
        fig_scalar, _ = plot_scalars(scalar_values[net_id])
    
    if verbose != None:
        set_log_level(current_level)
    
    return fig_scalar, fig_dist

def common_props_dict(networks):
    new = defaultdict(lambda:defaultdict())

    for i, (net_id, props) in enumerate(networks.items()):
        if i == 0:
            common = set(props.keys())
        else:
            common.intersection_update(set(props.keys()))

    new = {
        net_id : {
            prop_name : value 
            for prop_name, value in props.items()
            if prop_name in common
        }
        for net_id, props in networks.items()
    }

    return new

# Comparison of multiple networks
def compare_structure(
    networks: dict,
    norm: str | None = None,
    selected_props: str | list = "all",
    workers: str | int = "auto",
    include_env: None | dict = None,
    return_prop_dicts: bool = False,
    association_metric: Callable = pearsonr,
    verbose: str = None,
    erdos_renyi : int = 0
) -> Tuple[dict, dict] | plt.Figure:
    """
    Module-level function to compare multiple networks.

    Returns a tuple of figures, one for the scalar properties and one for the distributions if return_prop_dicts is False.
    Otherwise, it returns a tuple of dictionaries, one for the scalar properties and one for the distributions.

    Args:
        networks (dict): Dictionary of networks to compare.
            {'net_id': DiGraph | Graph}
        norm (str, optional): Normalization to apply. Defaults to None.
            Valid values are 'network', 'biological' or None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
        workers (int, optional): Number of workers to use. Defaults to 'auto'.
            Auto means number of cpu's - 1.

    Raises:
        NormalizationError: Raised if the normalization is not valid.
        ValueError: Raised if there is not enough data to compare.
    """
    if norm not in NORM_OPTIONS:
        struct_logger.critical("Normalization not valid")
        raise properties.NormalizationError("Normalization not valid")

    # handle workers
    usable_workers = cpu_count() - 1
    if workers == "auto":
        workers = usable_workers
    elif workers > usable_workers:
        struct_logger.warning(
            f"{workers} workers requested, but only {usable_workers} are available. Using {usable_workers} workers instead."
        )
        workers = usable_workers

    # currently, both selected_props and child_classes are being passed to get_props, however, only one is needed.
    # TODO: Optimization:  passing only child_classes would be more efficient beacuse it computes get_child_classes only once.
    child_classes = get_child_classes(PARENT_CLASS, selected_props, include_env=include_env)
    print(child_classes)

    # prepare data
    data = [
        list(networks.values()),
        list(networks.keys()),
        [norm] * len(networks),
        [selected_props] * len(networks),
        [child_classes] * len(networks),
        [verbose] * len(networks),  # verbose
        [True] * len(networks),  # keep_names
    ]

    # run parallel
    struct_logger.info('Analayzing inputed networks...')
    results = run_parallel(characterize_network, data, workers, verbose=verbose, process='analysis of inputed networks')
    struct_logger.info('Finished computing properties for all networks.')
    name_scalars_array = results["scalars"]
    name_moments_arrays = results["distributions"]
    
    for net_id, prop in name_moments_arrays.items():
        for prop_name, values in prop.items():
            name_scalars_array[net_id][f'Average {prop_name}'] = values[0]
            name_scalars_array[net_id][f'Variation {prop_name}'] = values[1]
            name_scalars_array[net_id][f'Skewness {prop_name}'] = values[2]
            name_scalars_array[net_id][f'Kurtosis {prop_name}'] = values[3]
    
    if erdos_renyi:

        if erdos_renyi < 0:
            struct_logger.critical('Erdos-Renyi argument must be 0 or greater.')
            raise ValueError("erdos_renyi must be 0 or greater")
    
        struct_logger.info('Analyzing ER networks...')
        for net_id, net in networks.items():
            er_scalars_array, er_moments_arrays = er_nets_per_net_analysis(
                                                                G= net, 
                                                                net_id= net_id, 
                                                                norm= norm, 
                                                                erdos_renyi= erdos_renyi, 
                                                                selected_props= selected_props,
                                                                workers= workers,
                                                                include_env= include_env,
                                                            )
            name_scalars_array.update(er_scalars_array)
            name_moments_arrays.update(er_moments_arrays)
    
    if return_prop_dicts:
        return name_scalars_array, name_moments_arrays
    
    # TODO: Optimization:  only compute the common properties
    name_scalars_array = common_props_dict(name_scalars_array)

    struct_logger.info('Starting comparison and building symmetric heatmap...')
    # Scalar properties
    if len(name_scalars_array) > 0 and len(list(name_scalars_array.values())[0]) > 1:
        df = association(name_scalars_array, corr_func=association_metric)
        fig_scalar = create_symmetric_heatmap(df, title=f"Global properties")
    else:
        struct_logger.critical("Not enough data to compare.")
        raise ValueError("Not enough data to compare.")

    return fig_scalar
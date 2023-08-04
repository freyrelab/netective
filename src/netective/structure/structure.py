from __future__ import annotations

import os
import math
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
from multiprocessing import cpu_count

from freyrelab.regnets.regnet import RegNet
from freyrelab.nets.paths2 import Efficiency, ShortestDistances, ShortestPaths

from netective import properties
from netective.properties import remove_self_loops
from netective.utils import compute_moments, run_parallel, validate_network, concat_path
from netective.structure.dataviz import plot_scalars, create_symmetric_heatmap, plot_distributions

import matplotlib.pyplot as plt

# Constants
NORM_OPTIONS = [None, "network", "biological"]
PARENT_CLASS = properties._Property

# Auxiliar Fxns
def flatten_list_of_iterables(lst):
    return list(chain.from_iterable(lst))


# Comparison Fxn
def pairwise_pearson_correlation(dict_data: dict[str, dict[str, float]]):
    # TODO: make this general for any correlation... # shouldn't be here...
    # Get the keys (name_dists) from the dictionary
    name_dists = list(dict_data.keys())

    # Initialize an empty DataFrame to store the correlation coefficients
    corr_df = pd.DataFrame(index=name_dists, columns=name_dists)

    # Calculate the pairwise Pearson correlation coefficients
    for i in range(len(name_dists)):
        for j in range(i, len(name_dists)):
            name_dist1 = name_dists[i]
            name_dist2 = name_dists[j]
            array1 = np.asarray(list(dict_data[name_dist1].values()))
            array2 = np.asarray(list(dict_data[name_dist2].values()))

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


# Get properties selected Fxn
def get_child_classes(parent_class, selected_props):
    child_classes = []
    all_properties = []
    print(f"Properties used for analysis: ", end=" ")
    if selected_props == "all":
        for name, obj in inspect.getmembers(properties):
            if inspect.isclass(obj) and issubclass(obj, parent_class) and obj != parent_class:
                print(obj.CLASS_NAME, end=" ")
                child_classes.append(obj)
                all_properties.append(obj.CLASS_NAME)
    else:
        for name, obj in inspect.getmembers(properties):
            if (
                inspect.isclass(obj)
                and issubclass(obj, parent_class)
                and obj != parent_class
                and name in selected_props
            ):
                print(name, end=" ")
                child_classes.append(obj)
                all_properties.append(obj.CLASS_NAME)
            if (
                inspect.isclass(obj)
                and issubclass(obj, parent_class)
                and obj != parent_class
                and obj.CLASS_NAME not in all_properties
            ):
                all_properties.append(obj.CLASS_NAME)
    print("\n")
    if len(child_classes) == 0:
        raise Exception(
            f"Sorry, no matches for properties inquired.\nList of available properties is: {all_properties}"
        )
    return child_classes


def _max_loops(n: int, r: int, tfs: int, r_tfs: int) -> int:
    """
    Computes the maximum number of motifs of size r with r_tfs TFs in a network of n nodes with tfs TFs.

    Args:
        n = number of nodes in the network
        r = number of elements in the motif
        tfs = number of TFs in the network
        r_tfs = number of TFs in the motif

    Returns:
        int: maximum number of motifs of size r with r_tfs TFs in a network of n nodes with tfs TFs.
    """
    putative = math.factorial(n) / math.factorial(n - r)
    # fraccion de TFs al cuadrado para feed forward (el expoente es el numero de TFs en el motivo)
    putative = putative * (
        (tfs / n) ** r_tfs
    )  # TODO: check this line to consider there shouldn't be replacement.
    return putative


class GraphObserver:

    """A class to observe changes in a graph."""

    def __init__(self, G: Graph | RegNet, data: bool = False) -> None:
        """
        Initialize the GraphObserver class.

        Args:
            G (nx.Graph | rn.RegNet): The graph to observe.
            data (bool): If True, the node/edge data will be considered when computing the hash.

        """
        self.G = G
        self.data = data
        self.graph_hash = self._compute_hash(self.G)

    def _compute_hash(self, G: Graph | RegNet) -> str:
        """
        Compute the hash of the graph.

        Args:
            G (nx.Graph | rn.RegNet): The graph to compute the hash.
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

    def changed(self, G: Graph | RegNet = None, update_G: bool = False) -> bool:
        """
        Check if G has changed with reference to the last call.

        Args:
            G (nx.Graph | rn.RegNet): The graph to check. If None, the original graph will be used.
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
        G: DiGraph | RegNet,
        norm: None | str | pd.Series = None,
        net_id: str = None,
        verbose: bool = False,
    ):

        """
        Object to compute, normalize, and store the structural properties of a network.
        Structure inherits from pandas.Series.

        Creating a Structure object does not compute the structural properties.
        Insted, it sets the attributes of the object for future property computation.
        Use the get_props() method to compute the structural properties.
        If the network has changed, the properties and the normalization factors are recomputed when get_props() is called.

        Args:
            G: DiGraph or RegNet.
                Network to compute the structural properties.
            norm: None|str|pd.Series.
                Normalization factor for each property.
                Use None to disable normalization.
                Use 'biol' to normalize by the biological scale factors.
                Use a dictionary to normalize by custom scale factors.
                Missing properties are reported as NaN.
            net_id: str.
                Name of the network. If None, a random uuid is assigned.
                Used for verbose mode and raising errors.
            verbose: bool.
                Whether to print information about the network.

        Returns:
            Structure: Object with the structural properties of the network.

        Raises:
            TypeError: If G is not a DiGraph or a RegNet.
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
        print(f"init --- graph_observer.graph_hash: {self.graph_observer.graph_hash}")
        self.norm_observer = NormObserver(
            norm
        )  # object to observe changes in the normalization strategy
        self.verbose = verbose
        self.net_id = net_id if net_id is not None else str(uuid.uuid4())

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

        if self.verbose:
            print("Normalizing...", flush=True)

        norm_scalar_values = {}
        norm_dist_values = {}
        for name, x in instances.items():
            # TODO: why using x.CLASS_NAME instead of name?
            # TODO: toma la normalización de normalizaciones disponibles
            dict_ = norm_scalar_values if x._return_type == "scalar" else norm_dist_values
            if norm not in NORM_OPTIONS:
                raise properties.NormalizationError(f"Invalid normalization method: {norm}")
            try:
                if norm == "network":
                    dict_[x.CLASS_NAME] = x.norm_network()
                elif norm == "biological":
                    dict_[x.CLASS_NAME] = x.norm_biol()
            except (NotImplementedError, properties.NormalizationError):
                dict_[x.CLASS_NAME] = np.nan

        return norm_scalar_values, norm_dist_values

    def _compute_props(self, child_classes) -> dict[str, float, int]:

        """
        Computes the structural properties of a network.

        Args:
            G: DiGraph or RegNet.
                Network to compute the structural properties.
            norm: None|str|pd.Series.
                Normalization factor for each property.
            net_id: str.
                Name of the network. If None, a random uuid is assigned.
            child_classes: list.
                List of classes to compute the structural properties.

        Returns:
            dict: Dictionary with the structural properties of the network.
        """

        if self.verbose:
            print(f"Processing {self.net_id}...", flush=True)
            print(
                f"{self.net_id} has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.",
                flush=True,
            )

        # Get Instances Fxns
        def get_instances_no_paths(H, child_classes):
            instances = {x.CLASS_NAME: x(H.copy()) for x in child_classes if not x._use_paths}
            return instances

        def get_instances_paths(H, child_classes, net_shortest_paths, net_shortest_distances):
            instances = {
                x.CLASS_NAME: x(
                    H.copy(),
                    net_shortest_paths=net_shortest_paths,
                    net_shortest_distances=net_shortest_distances,
                )
                for x in child_classes
                if x._use_paths
            }
            return instances

        # Properties that do not use paths object
        instances = get_instances_no_paths(self.G, child_classes)

        # TODO!!! agregar instancias para redes sin self-loops y direccion. Revisar si conviene hacer compbinatoria.
        # Paths objects
        remove_self_loops(self.G)
        self.G = self.G.giant_component
        self.G = self.G.to_undirected()
        net_shortest_paths = ShortestPaths(self.G)
        net_shortest_distances = ShortestDistances(self.G)
        self.dist_values = {}

        # Properties that use paths object
        # They use the giant component from an undirected graph with no selfloops
        instances.update(
            get_instances_paths(self.G, child_classes, net_shortest_paths, net_shortest_distances)
        )

        self.scalar_values = {
            x.CLASS_NAME: x.compute() for name, x in instances.items() if x._return_type == "scalar"
        }

        self.dist_values = {
            x.CLASS_NAME: x.compute()
            for name, x in instances.items()
            if x._return_type == "distribution"
        }

        if self.norm_observer.norm is not None:
            self.scalar_values, self.dist_values = self._normalize_props(
                instances, norm=self.norm_observer.norm
            )

        self.dist_values = {k: v for k, v in self.dist_values.items() if not np.isnan(v).all()}

        return self.scalar_values, self.dist_values

    def get_props(
        self, props: str | list = "all", child_classes: list = None
    ) -> Tuple[dict[str, dict], dict[str, dict]]:
        """
        Computes the structural properties of a network.
        Either props or child_classes must be provided.
        If both are provided, props is ignored.

        Args:
            props: str|list.
                Structural properties to compute.
                Use 'all' to compute all the properties.
                Use a list of property names to compute only those properties.
                Use a list of property classes to compute all the properties of those classes.
            child_classes: list.
                List of property classes to compute all the properties of those classes.


        Returns:
            If keep_names is True, it returns a dictionary with the property names as keys.
            Otherwise, it returns a flattened array with the values.
        """
        # Compute the properties if they have not been computed yet or if the network has changed
        # print(
        #         f", is None?: {self.graph_observer.graph_hash is None}, graph changed: {self.graph_observer.changed(self._original_G, update_G=True)}, norm changed: {self.norm_observer.change()}"
        #     )
        if (
            self.graph_observer.graph_hash is None
            or self.graph_observer.changed(self._original_G, update_G=True)
        ) or self.norm_observer.change():
            if self.verbose:
                print(
                    "The network or the normalization method has changed. Computing its properties...",
                    flush=True,
                )
                # TODO!!!: conservar los raw!
                # TODO: incluir un verbose cuando semodifica la normalización. O usar en su lugar un warning?

            # TODO!!!!: quita parche, primera corrida es None y self.graph_observer.changed(self._original_G no entra por bypass.
            if self.graph_observer.graph_hash is None:
                self.graph_observer.changed(self._original_G, update_G=True)

            print(f"hash: {self.graph_observer.graph_hash}")
            self.G = validate_network(
                self._original_G.copy()
            )  # to make sure the network is valid and use the actual modified network
            try:
                self._scalar_arrays = {}
                self._dist_moments_arrays = {}

                if child_classes is None:
                    child_classes = get_child_classes(PARENT_CLASS, props)

                # props
                scalar_values, dist_values = self._compute_props(child_classes)

                self._scalar_arrays[self.net_id] = scalar_values
                self._dist_moments_arrays[self.net_id] = {
                    prop_name: compute_moments(array) for prop_name, array in dist_values.items()
                }

                return self._scalar_arrays, self._dist_moments_arrays

            # This is a general exception handler to catch any error that may occur in the parallelized code
            except Exception as e:
                tracebackString = traceback.format_exc(e)
                raise NotImplementedError(
                    f"\n\nError occurred. Original traceback is\n{tracebackString}\n"
                )

        return self._scalar_arrays, self._dist_moments_arrays


def struc_props_call(
    G: DiGraph | RegNet,
    net_id: str,
    norm: None | str | pd.Series,
    erdos_renyi: int,
    verbose: bool = False,
) -> list(tuple(str, dict)):

    """
    Call the function struc_props with erdos_renyi random graphs with the same number of nodes and edges as G.

    Args:
        G: DiGraph or RegNet.
            Network to compute the structural properties.
        net_id: str.
            Name of the network.
        norm: bool.
            If True, the properties are normalized (biological criteria).
        erdos_renyi: int.
            Number of random graphs to generate with the same number of nodes and edges as G.
            If 0, only the properties of G are computed.
            If greater than 0, the properties of G and the average properties of the random graphs are computed.

    Returns:
        list(tuple(str, dict)): list of tuples with the network id and the properties of the network.

    Raises:
        ValueError: if erdos_renyi is less than 0.
    """
    # TODO!! UPDATE
    S_bio = Structure(G, norm=norm, net_id=net_id, verbose=verbose)
    props = S_bio.get_props()

    if erdos_renyi < 0:
        raise ValueError("erdos_renyi must be 0 or greater")

    elif erdos_renyi == 0:
        netid_props = [(net_id, props)]

    else:
        from collections import defaultdict
        from networkx import fast_gnp_random_graph

        props_er = defaultdict(list)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        for i in range(erdos_renyi):
            ER = RegNet(fast_gnp_random_graph(n, m / (n**2), directed=True))
            S_er = Structure(ER, norm=norm, net_id=f"{net_id}_ER_{i}", verbose=verbose)
            props_i = S_er.get_props()
            for k, v in props_i.items():
                props_er[k].append(v)

        # TODO: here you can make it robust to a threshold of nan values
        props_er_avg = {prop: sum(vals) / len(vals) for prop, vals in props_er.items()}
        netid_props = [(net_id, props), (f"{net_id}_ER_avg", props_er_avg)]

    return netid_props


def save_strucs(
    fig_scalar,
    fig_dist,
    output: str = os.getcwd(),
    delimiter: str = "\t",
    cl: str = None,
    output_file: str = "structural_properties",
) -> None:

    # TODO!! UPDATE

    """
    Save the structural properties in a file.

    Args:
        df: DataFrame.
            Dataframe with the structural properties.
        output: str.
            Path to the output directory. Default is the current working directory.
        delimiter: str.
            Delimiter to use in the output file. Default is tab.
        cl: str.
            Command line used to run the script. Default is None.
        output_file: str.
            Name of the output file. Default is structural_properties.

    Returns:
        None.
    """

    exts = {",": "csv", "\t": "tsv"}
    ext = exts.get(delimiter, "txt")
    # file_p = concat_path(output, f"{output_file}.{ext}")

    file_p_s = concat_path(output, "network_level_comp.png")
    file_p_d = concat_path(output, "node_level_comp.png")

    # save output
    # rewrite file if it exists
    # with open(file_p, "w") as f:
    # save command line
    # print(cl, file=f)

    # save structural properties
    fig_scalar.savefig(file_p_s, dpi=300)
    fig_dist.savefig(file_p_d, dpi=300)


# User Fxns
# Characterization of one network
def characterize_network(
    G: RegNet,
    name: str,
    norm: str | None = None,
    selected_props: str | list = "all",
    child_classes: list = None,
    verbose: bool = False,
    return_prop_dicts: bool = False,
) -> None | Tuple[dict, dict]:
    """Module-level function to characterize a single network.

    Args:
        G (RegNet): Network to characterize.
        norm (str, optional): Normalization to apply. Valid values are 'network', 'biological' or None. Defaults to None.
        selected_props (str | list, optional): Properties to compute. Defaults to 'all' (all properties).
        child_classes (list, optional): List of child classes to compute. Defaults to None. Use either selected_props or child_classes.
            if child_classes is not None, selected_props is ignored.
        verbose (bool, optional): If True, print messages. Defaults to False.

    Returns:
        dict: Dictionary with the properties of the network if return_prop_dicts is True.
        None: If return_prop_dicts is False. The figures are shown.

    Raises:
        Exception: Raised if the normalization is not valid.

    """

    struc = Structure(G, norm=norm, net_id=name, verbose=verbose)
    if child_classes is not None:
        scalar_values, dist_values = struc.get_props(child_classes=child_classes)
    else:
        scalar_values, dist_values = struc.get_props(props=selected_props)

    if len(dist_values) == 0 and len(scalar_values) == 0:
        raise ValueError("Not enough data, try with more properties or another normalization")

    print(f"1:   {scalar_values}")

    if return_prop_dicts:
        return scalar_values, dist_values

    print(f"2:   {dist_values}")
    if len(dist_values) != 0:
        fig_dist, _ = plot_distributions(dist_values[name])
        fig_dist.show()
    if len(scalar_values) != 0:
        print(scalar_values)
        fig_scalar, _ = plot_scalars(scalar_values[name])
        fig_scalar.show()


# Comparison of multiple networks
def compare_networks(
    networks: dict,
    norm: str | None = None,
    selected_props: str | list = "all",
    workers: str | int = "auto",
    return_prop_dicts: bool = False,
) -> Tuple[dict, dict] | Tuple[plt.Figure, plt.Figure]:

    """Module-level function to compare multiple networks.

    Returns a tuple of figures, one for the scalar properties and one for the distributions if return_prop_dicts is False.
    Otherwise, it returns a tuple of dictionaries, one for the scalar properties and one for the distributions.

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
    # TODO!!: passing only child_classes would be more efficient beacuse it computes get_child_classes only once.
    child_classes = get_child_classes(PARENT_CLASS, selected_props)

    # prepare data
    data = [
        list(networks.values()),
        list(networks.keys()),
        [norm] * len(networks),
        [selected_props] * len(networks),
        [child_classes] * len(networks),
        [False] * len(networks),  # verbose
        [True] * len(networks),  # keep_names
    ]

    # run parallel
    results = run_parallel(characterize_network, data, workers)
    name_scalars_array = results["scalars"]
    name_moments_arrays = results["distributions"]

    if return_prop_dicts:
        return name_scalars_array, name_moments_arrays

    # Scalar properties
    if len(name_scalars_array) > 0 and len(list(name_scalars_array.values())[0]) > 1:
        df = pairwise_pearson_correlation(name_scalars_array)
        fig_scalar, _ = create_symmetric_heatmap(df, title=f"Global properties")
    else:
        print(name_scalars_array)
        raise ValueError("Not enough data to compare.")

    # Distribution properties
    if len(name_moments_arrays) > 0 and len(list(name_moments_arrays.values())[0]) > 1:
        df = pairwise_pearson_correlation(name_moments_arrays)
        fig_dist, _ = create_symmetric_heatmap(df, title=f"Local properties")
    else:
        raise ValueError("Not enough data to compare.")

    return fig_scalar, fig_dist

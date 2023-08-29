from __future__ import annotations

import math
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
import igraph as ig

from netective.utils import Efficiency, giant_component_size

# Decorators for graph characteristics

def use_direction(cls):
    cls._use_direction = True
    return cls


def use_selfloops(cls):
    cls._use_selfloops = True
    return cls


def use_giant_component(cls):
    cls._use_giant_component = True
    return cls


def return_scalar(cls):
    cls._return_type = "scalar"
    return cls


def return_distribution(cls):
    cls._return_type = "distribution"
    return cls


def use_paths(cls):
    cls._use_paths = True
    return cls


def validate_network_characteristics(self):

    directed = self.G.is_directed()
    giant_components = True if nx.number_connected_components(self.G) == 1 else False
    paths = self._use_paths

    if self._use_selfloops:
        property_mask = np.asarray(
            [self._use_direction, self._use_giant_component, self._use_paths]
        )
        input_net_mask = np.asarray([directed, giant_components, paths])
    else:
        property_mask = np.asarray(
            [self._use_selfloops, self._use_direction, self._use_giant_component, self._use_paths]
        )
        selfloops = True if nx.number_of_selfloops(self.G) > 1 else False
        input_net_mask = np.asarray([selfloops, directed, giant_components, paths])

    if not np.array_equal(property_mask, input_net_mask):
        raise ("Error, incorrect format of input network for this propertie's computation.")


# Theoretical maximums
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
    putative = putative * (
        (tfs / n) ** r_tfs
    )  # TODO: check this line to consider there shouldn't be replacement.
    return putative


# Auxiliary functions
def remove_self_loops(G: nx.DiGraph):
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def get_entropy(elements: np.array):
    """Get the entropy of an array of elements."""
    entropy = (elements * np.log2(elements)).sum()
    return entropy if entropy == 0 else -entropy


def get_parent_nodes(G: nx.DiGraph):
    """Get the parent nodes of a graph."""
    return [i for i, k_out in G.out_degree() if k_out > 0]

# motifs class

class count_3motifs:
    """Summary."""
    def __init__(self, G):
        if not G.is_directed():
            raise TypeError("requires a directed graph")
        iG = ig.Graph.TupleList(G.edges(data=False), directed=True, vertex_name_attr='name', edge_attrs=None, weights=False)
        iG.add_vertices(nx.isolates(G))
        self.tc = iG.triad_census()
    
    @property
    def feedforwards(self):
        """Summary."""
        return self.tc.t030T
    
    @property
    def complex_feedforwards(self):
        """Summary."""
        return self.tc.t120U
    
    @property
    def feedbacks(self):
        """Summary."""
        return self.tc.t030C + self.tc.t120C + self.tc.t210 + 2 * self.tc.t300

# Parent class for all properties

class _Property(ABC):
    """Abstract base class for all properties."""

    _return_type = None
    _use_paths = False
    _use_direction = False
    _use_selfloops = False
    _use_giant_component = False

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self._raw_value = None
        self._n_nodes = self.G.number_of_nodes()
        if self._n_nodes == 0:
            raise NullGraphError("A null graph has no self-regulations.")

    @abstractmethod
    def compute(self):
        return self._raw_value

    @abstractmethod
    def norm_biol(self, *args):
        pass

    @abstractmethod
    def norm_network(self, *args):
        pass


# Error handling
class NormalizationError(Exception):
    """Exception raised for errors in the normalization.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="An error occurred during normalization."):
        self.message = message
        super().__init__(self.message)


class NullGraphError(Exception):
    """Exception raised for null graph."""

    pass


class EmptyGraphError(Exception):
    """Exception raised for empty graph. Nodes with no edges."""

    pass


def check_raw_value(func):
    """Decorator to check if raw value is None. If it is, raise an error."""

    def wrapper(self, *args):
        if self._raw_value is not None:
            return func(self, *args)
        else:
            raise ValueError("Raw value is None. Call compute() method first.")

    return wrapper


# Properties
@return_scalar
@use_direction
@use_selfloops
@use_giant_component
class Density(_Property):
    """Density of the graph.

    The density of a graph is defined as the ratio of the number of edges to the number of possible edges.
    The number of possible edges for directed network with self loops is given by n**2,
    where n is the number of nodes in the graph.

    Methods:
        compute: Compute the density of the graph.
        norm_biol: Normalize the density of the graph to the number of parents.
        norm_network: Normalize the density of the graph to the number of nodes.
    """

    CLASS_NAME = "Density"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Directed graph.
        """
        super().__init__(G)

    def compute(self) -> float:
        """Compute the density of the graph.

        Returns:
            float: Density of the graph.
        """
        n_edges = self.G.number_of_edges()
        self._raw_value = n_edges / self._n_nodes**2
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the density of the graph to the number of parents.

        Returns:
            float: Normalized density of the graph, considering the number of parents.

        Raises:
            NormalizationError: If the graph has no parent nodes.
        """
        n_parents = len(get_parent_nodes(self.G))
        try:
            return self._raw_value * (self._n_nodes / n_parents)
        except ZeroDivisionError:
            raise NormalizationError(
                "Division by zero (no parent nodes). Cannot normalize with this approach."
            )

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the density of the graph to the number of nodes. (Already normalized)

        Returns:
            float: Normalized density of the graph, considering the number of nodes.
        """
        return self._raw_value  # density is already normalized to [0,1]


@return_scalar
@use_direction
@use_selfloops
class Regulators(_Property):
    """Number of regulators of the graph.

    Methods:
        compute: Compute the number of regulators of the graph.
        norm_biol: Normalize the number of regulators of the graph to the number of parents.
        norm_network: Normalize the number of regulators of the graph to the number of nodes.
    """

    CLASS_NAME = "Regulators"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Directed graph.
        """
        super().__init__(G)

    def compute(self) -> int:
        """Compute the number of regulators of the graph.

        Returns:
            int: Number of regulators of the graph.
        """
        self._raw_value = len(get_parent_nodes(self.G))
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the number of regulators of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the number of regulators of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes


@return_scalar
@use_direction
@use_selfloops
class SelfRegulations(_Property):
    """Number of self-regulations of the graph.

    Methods:
        compute: Compute the number of self-regulations of the graph.
        norm_biol: Normalize the number of self-regulations of the graph to the number of parents.
        norm_network: Normalize the number of self-regulations of the graph to the number of nodes.
    """

    CLASS_NAME = "Self-Regulations"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Directed graph.
        """
        super().__init__(G)

    def compute(self) -> int:
        """Compute the number of self-regulations of the graph.

        Returns:
            int: Number of self-regulations of the graph.
        """
        self._raw_value = nx.number_of_selfloops(self.G)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the number of self-regulations of the graph to the number of parents."""
        n_parents = len(get_parent_nodes(self.G))
        try:
            return self._raw_value / n_parents
        except ZeroDivisionError:
            raise NormalizationError(
                "Division by zero (no parent nodes). Cannot normalize with this approach."
            )

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the number of self-regulations of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes


@return_scalar
@use_direction
@use_selfloops
class MaxOutDegree(_Property):
    """Maximum out-degree of the graph.

    Methods:
        compute: Compute the maximum out-degree of the graph.
        norm_biol: Normalize the maximum out-degree of the graph to the number of parents.
        norm_network: Normalize the maximum out-degree of the graph to the number of nodes.
    """

    CLASS_NAME = "Max Out-Degree"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        """Compute the maximum out-degree of the graph.

        Returns:
            int: Maximum out-degree of the graph.
        """
        self._raw_value = max(dict(self.G.out_degree()).values())
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the maximum out-degree of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the maximum out-degree of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes


@return_scalar
@use_direction
@use_selfloops
class MaxInDegree(_Property):
    """Maximum in-degree of the graph.

    Methods:
        compute: Compute the maximum in-degree of the graph.
        norm_biol: Normalize the maximum in-degree of the graph to the number of parents.
        norm_network: Normalize the maximum in-degree of the graph to the number of nodes.
    """

    CLASS_NAME = "Max In-Dregree"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        """Compute the maximum in-degree of the graph.

        Returns:
            int: Maximum in-degree of the graph.
        """
        self._raw_value = max(dict(self.G.in_degree()).values())
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the maximum in-degree of the graph to the number of parents."""
        n_parents = len(get_parent_nodes(self.G))
        try:
            return self._raw_value / n_parents
        except ZeroDivisionError:
            raise NormalizationError(
                "Division by zero (no parent nodes). Cannot normalize with this approach."
            )

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the maximum in-degree of the graph to the number of nodes."""
        return self._raw_value / self._n_nodes


@return_scalar
@use_direction
class FeedbackLoops_3(_Property):
    """Number of feedback loops of length 3.

    Methods:
        compute: Compute the number of feedback loops of length 3.
        norm_biol: Normalize the number of feedback loops of length 3 to the number of parents.
        norm_network: Normalize the number of feedback loops of length 3 to the number of nodes.
    """

    CLASS_NAME = "3-Feedback Loops"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        mc = count_3motifs(self.G)
        self._raw_value = mc.feedbacks
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the number of feedback loops of length 3 to the number of parents."""
        n_parents = len(get_parent_nodes(self.G))
        max_feedbacks3 = (
            n_parents * (n_parents - 1) * (n_parents - 2)
        )  # TODO: UERGENTE is this better than _max_loops???
        return self._raw_value / max_feedbacks3

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the number of feedback loops of length 3 to the number of nodes."""
        max_feedbacks3 = self._n_nodes * (self._n_nodes - 1) * (self._n_nodes - 2)  # TODO: verify
        return self._raw_value / max_feedbacks3


# TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! INCLUIR feedforwards_count
@return_scalar
@use_direction
class ComplexFeedForwardCircuits(_Property):
    """Number of complex feed-forward circuits.

    Methods:
        compute: Compute the number of complex feed-forward circuits.
        norm_biol: Normalize the number of complex feed-forward circuits to the number of parents.
        norm_network: Normalize the number of complex feed-forward circuits to the number of nodes.
    """

    CLASS_NAME = "Complex Feed-Forward Circuits"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        mc = count_3motifs(self.G)
        self._raw_value = mc.complex_feedforwards
        return self._raw_value

    def norm_biol(self) -> float:
        """Normalize the number of complex feed-forward circuits to the number of parents."""
        n_parents = len(get_parent_nodes(self.G))
        max_complex_ff = n_parents * (n_parents - 1)  # TODO: verify
        return self._raw_value / max_complex_ff

    def norm_network(self) -> float:
        """Normalize the number of complex feed-forward circuits to the number of nodes."""
        max_complex_ff = self._n_nodes * (self._n_nodes - 1) * (self._n_nodes - 2)  # TODO: verify
        return self._raw_value / max_complex_ff


@return_scalar
@use_giant_component
class GenesintheGiantComponent(_Property):
    """Number of genes in the giant component.

    Methods:
        compute: Compute the number of genes in the giant component.
        norm_biol: Normalize the number of genes in the giant component to the number of parents.
        norm_network: Normalize the number of genes in the giant component to the number of nodes.
    """

    CLASS_NAME = "Gene % in the Giant Component"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = giant_component_size(self.G)

        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the number of genes in the giant component to the number of nodes."""
        return self._raw_value / self._n_nodes

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the number of genes in the giant component to the number of nodes."""
        return self._raw_value / self._n_nodes


@use_paths
@return_scalar
@use_giant_component
class Diameter(_Property):

    """Diameter of the graph.

    Methods:
        compute: Compute the diameter of the graph using the giant component.
        norm_biol: Normalize the diameter of the graph to the number of parents.
        norm_network: Normalize the diameter of the graph to the number of nodes.
    """

    CLASS_NAME = "Diameter"

    def __init__(self, G: nx.DiGraph, **kwargs):
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = self._shortest_distances.diameter
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the diameter of the graph to the number of parents in the giant component.

        The maximum diameter of a graph with n nodes is n-1.
        When only considering the parents, the maximum diameter is n_parents.
        Considering both, the maximum diameter is n_parents (every parent, then a leaf node).

        Note: the diameter of the giant component is not necessarily the same as the diameter of the graph.
        we are using the diameter of the giant component here given that the computation of the diameter is performed on the giant component.
        """
        """
        n_parents = len(get_parent_nodes(self.G))
        return self._raw_value / n_parents
        """
        raise NotImplementedError

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the diameter of the graph to the number of nodes.

        The maximum diameter of a graph with n nodes is n-1.
        """
        return self._raw_value / (self._n_nodes - 1)


@use_paths
@return_scalar
@use_giant_component
class AverageShortestPathLength(_Property):
    """Average shortest path length of the graph.

    Methods:
        compute: Compute the average shortest path length of the graph using the giant component.
        norm_biol: Normalize the average shortest path length of the graph to the number of parents.
        norm_network: Normalize the average shortest path length of the graph to the number of nodes.
    """

    CLASS_NAME = "Average Shortest Path Length"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> int:
        """Compute the average shortest path length of the graph using the giant component.

        Returns:
            int: Average shortest path length of the graph.
        """
        self._raw_value = self._shortest_distances.average_path_length
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        # """Normalize the average shortest path length of the graph to the number of parents in the giant component."""
        # n_parents = len(get_parent_nodes(self.G))
        # return self._raw_value / n_parents
        raise NotImplementedError

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the average shortest path length of the graph to the number of nodes."""
        return self._raw_value / (self._n_nodes - 1)


@return_scalar
class AverageClusteringCoefficient(_Property):
    """Average clustering coefficient of the graph.

    Methods:
        compute: Compute the average clustering coefficient of the graph.
        norm_biol: Normalize the average clustering coefficient of the graph to the number of parents.
        norm_network: Normalize the average clustering coefficient of the graph to the number of nodes.
    """

    CLASS_NAME = "Average Clustering Coefficient"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> float:
        # self._raw_value = self.G.average_clustering_coefficient
        clustering_values = nx.clustering(self.G).values()
        self._raw_value = np.fromiter(clustering_values, dtype=float).mean()
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Coeffients are considered already normalized."""
        return self._raw_value

    @check_raw_value
    def norm_network(self) -> float:
        """Coeffients are considered already normalized."""
        return self._raw_value


@return_distribution
class ClusteringCoefficient(_Property):
    """Clustering coefficient of the graph.

    Methods:
        compute: Compute the clustering coefficient of the graph.
    """

    CLASS_NAME = "Clustering Coefficient"

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> float:
        clustering_values = nx.clustering(self.G).values()
        self._raw_value = np.fromiter(clustering_values, dtype=float)
        return self._raw_value

    def norm_biol(self) -> float:
        """Coeffients are considered already normalized."""
        raise NotImplementedError

    def norm_network(self) -> float:
        """Coeffients are considered already normalized."""
        return (
            self._raw_value
        )  # TODO: o cambiar a NotImplementedError??? El usuario debería de decidir qué hacer en este caso, dejar nan o el valor crudo.


@return_distribution
@use_direction
@use_selfloops
class InDegree(_Property):
    """In degree of the graph.

    The in degree of each node is defined as the number of predecessors it has.

    Methods:
        compute: Compute the in connectivity of the graph.
        norm_biol: Normalize the in connectivity of the graph to the number of parents.
        norm_network: Normalize the in connectivity of the graph to the number of nodes.
    """

    CLASS_NAME = "In-Degree"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the in-degree of the graph.

        Returns:
            nparray: Array with in-degrees of every node in graph.
        """
        a = [x for a, x in self.G.in_degree()]
        self._raw_value = np.array(a)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the nparray with the in-degrees of the graph to the number of parents"""
        n_parents = len(get_parent_nodes(self.G))
        try:
            return self._raw_value * (1 / n_parents)
        except ZeroDivisionError:
            raise NormalizationError(
                "Division by zero (no parent nodes). Cannot normalize with this approach."
            )

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the nparray with the in-degrees of the graph to the number of nodes."""
        return self._raw_value * (1 / self._n_nodes)


@return_distribution
@use_direction
@use_selfloops
class OutDegree(_Property):
    """Out degree of the graph.

    The out degree of each node is defined as the number of successors it has.

    Methods:
        compute: Compute the out connectivity of the graph.
        norm_biol: Normalize the out connectivity of the graph to the number of nodes.
        norm_network: Normalize the out connectivity of the graph to the number of nodes.
    """

    CLASS_NAME = "Out-Degree"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the out-degree of the graph.

        Returns:
            nparray: Array with out-degrees of every node in graph.
        """
        self._raw_value = np.array([x for a, x in self.G.out_degree()])
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the nparray with the out-degrees of the graph to the number of nodes"""
        return self._raw_value * (1 / self._n_nodes)

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the nparray with the out-degrees of the graph to the number of nodes."""
        return self._raw_value * (1 / self._n_nodes)


# TODO Por laguna perra razón se necesita hacer el to_undirected dentro de aquí????????
@return_distribution
class RichClub(_Property):
    """Rich Club Coefficient.

    The Rich Club Coefficient for every degree in the graph is defined as the clustering coeffficient
    of nodes with a higher degree than the degree being evaluated.

    Methods:
        compute: Compute the rich club coefficient of the graph.
        norm_biol: NO IMPLEMENTATION.
        norm_network: NO IMPLEMENTATION.
    """

    CLASS_NAME = "Rich Club Coefficient"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)
        self.A = self.G.to_undirected()

    def compute(self) -> np.array:
        """Compute the Rich Club Coefficient.

        Returns:
            np.array: distribution of rich club coefficients, by degree in graph.
        """
        n_edges = self.A.number_of_edges()
        if n_edges == 0:
            raise EmptyGraphError("There are no edges. Can not form rich clubs with no edges.")

        dict_coeff = nx.rich_club_coefficient(self.A, normalized=False)
        self._raw_value = np.fromiter(dict_coeff.values(), dtype=float)

        return self._raw_value

    @check_raw_value
    def norm_biol(self):
        raise NormalizationError("No biological normalization implemented.")

    @check_raw_value
    def norm_network(self):
        raise NormalizationError("No theoretical normalization implemented.")


@return_distribution
@use_selfloops
class SubgraphCentrality(_Property):
    """Subgraph centrality.

    Subgraph centrality is defined as the "sum" of closed walks of different lengths throught the network
    starting and ending in each node.

    Methods:
        compute: Compute the subgraph centrality for every node in the graph.
        norm_biol: NO IMPLEMENTATION.
        norm_network: Normalize subgraph centrality for every node to the max theoretical value.
    """

    CLASS_NAME = "Subgraph Centrality"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the subgraph centrality for every node in the graph.

        Returns:
            nparray: subgraph centrality for every node.
        """
        n_edges = self.G.number_of_edges()
        if n_edges == 0:
            raise EmptyGraphError(
                "There are no edges. Can not calculate subgraph centrality of nodes that do not form any edges."
            )
        self._raw_value = nx.subgraph_centrality(self.G)
        self._raw_value = np.fromiter(self._raw_value.values(), dtype=float)

        return self._raw_value

    @check_raw_value
    def norm_biol(self):
        raise NormalizationError("No biological normalization implemented.")

    @check_raw_value
    def norm_network(self):
        """Normalize the subgraph centrality of the graph to the max value, obtained from a complete graph of the same size"""

        T = nx.Graph()
        n_nodes = self._n_nodes
        T.add_nodes_from(range(n_nodes))
        T.add_edges_from([(i, j) for i in range(n_nodes) for j in range(n_nodes)])

        max = SubgraphCentrality(T)

        return self._raw_value / max.compute()


@return_distribution
@use_selfloops
class LocalityIndex(_Property):
    """Locality Index.

    Measurement that reflects the internality of every connection of all neighbors immediate to each node.
    It is calculated, for each node, as the number of links between that node's neighbors (including itself if there is a self-loop)
    divided by the total number of edges each of those neighbors is involved in.

    Methods:
        compute: Compute the locality index for every node in the graph.
        norm_biol: NO IMPLEMENTATION.
        norm_network: Normalize locality index for every node. Already normalized.
    """

    CLASS_NAME = "Locality Index"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the locality index for every node in the graph.

        Returns:
            nparray: locality index for every node.
        """
        n_edges = len(self.G.edges())
        # n_edges = self.G.number_of_edges()
        if n_edges == 0:
            raise EmptyGraphError(
                "There are no edges. Can not calculate locality index of nodes that do not form any edges."
            )

        self._raw_value = []

        for node in self.G.nodes():
            # If there is a self-loop, you can be your own neighbor, if not, a connection to you is considered in n_ext
            neighbors = [x for x in self.G.neighbors(node)]

            links_neighbors = [
                self.G.has_edge(neighbors[i], neighbors[j])
                for i in range(len(neighbors))
                for j in range(i, len(neighbors))
            ]

            n_int = np.fromiter(links_neighbors, dtype=int).sum()

            links_externals = [
                self.G.has_edge(i, j)
                for i in neighbors
                for x, j in self.G.edges(i)
                if j not in neighbors
            ]

            n_ext = np.fromiter(links_externals, dtype=int).sum()

            self._raw_value.append(n_int / (n_int + n_ext))

        self._raw_value = np.asarray(self._raw_value, dtype=float)

        return self._raw_value

    @check_raw_value
    def norm_biol(self):
        """NO IMPLEMENTATION."""
        raise NormalizationError("No biological normalization implemented.")

    @check_raw_value
    def norm_network(self):
        """Already normalized."""
        return self._raw_value  # Already normalized [0,1]


@return_distribution
@use_direction
class AverageOutDegreeNearestNeighbors(_Property):
    """Average Out-Degree of Nearest Neighbors

    Average out-degree of nearest neighbors is defined as de average of out-degrees of each memeber of a node's neighborhood.
    In this case, the neighborhood of each node is the list of its successors.

    Methods:
        compute: Compute the average out-degree for each node in the graph.
        norm_biol: Normalize the average out-degree of nearest neighbors to all nodes in the graph AND eliminate uninformative 0s.
        norm_network: Normalize the average out-degree of nearest neighbors to all nodes in the graph.
    """

    CLASS_NAME = "Average Degree for Nearest Neighbors (Out-Out)"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the average out-degree of nearest neighbors using Networkx implementation.

        Returns:
            np.array: average out-degree for each node in the graph.
        """
        self._dict_av_degree = nx.average_neighbor_degree(self.G, source="out", target="out")
        self._raw_value = np.fromiter(self._dict_av_degree.values(), dtype=float)

        return self._raw_value

    @check_raw_value  # Decorator to check if raw value is None. If it is, raise an error.
    def norm_biol(self) -> np.array:
        """Normalize the average degree of nearest neighbors for every node in the graph to the number of nodes and exclude
        0s from nodes that do not have a out-degree higher than 0. Relation between order of values and order of nodes is lost.
        """
        parents_value = np.array([self._dict_av_degree[node] for node in get_parent_nodes(self.G)])
        return parents_value * (1 / (self._n_nodes - 1))

    @check_raw_value
    def norm_network(self) -> np.array:
        """Normalize the average degree for nearest neighbors to the number of nodes (-1 because you can not be your own neighbor)"""
        return self._raw_value * (1 / (self._n_nodes - 1))


@return_distribution
class AverageDegreeNearestNeighbors(_Property):
    """Average Degree of Nearest Neighbors.

    The Average Degree of Nearest Neighbors here is considered for an undirected network, meaning the neighborhood for each node
    is all nodes it has a connection with, regardless of direction.

    Methods:
        compute: Compute the average degree of nearest neighbors.
        norm_biol: NO IMPLEMENTATION.
        norm_network: Normalize the average degree of nearest neighbors to all nodes in the graph.
    """

    CLASS_NAME = "Average Degree for Nearest Neighbors (Undirected)"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the density of the graph.

        Returns:
            float: Density of the graph.
        """
        dict_av_degree = nx.average_neighbor_degree(self.G)
        self._raw_value = np.fromiter(dict_av_degree.values(), dtype=float)

        return self._raw_value

    @check_raw_value  # Decorator to check if raw value is None. If it is, raise an error.
    def norm_biol(self):
        raise NormalizationError("No biological normalization implemented.")

    @check_raw_value
    def norm_network(self) -> np.array:
        """Normalize the average degree for nearest neighbors to the number of nodes (-1 because you can not be your own neighbor)"""
        return self._raw_value * (1 / (self._n_nodes - 1))


@return_scalar
@use_direction
@use_selfloops
class EntropyPKout(_Property):
    """Entropy of degree distribution.

    Entropy of the degree distribution is defined as the absolute valueof the sum of each probability to have a certain degree
    multiplied by its own log2.

    Methods:
        compute: Compute the entropy of the degree distribution for a graph.
        norm_biol: Normalize the entropy of the degree distribution to the max theoretical value.
        norm_network: Normalize the entropy of the degree of distribution to the max theoretical value.
    """

    CLASS_NAME = "Entropy of Degree Distribution"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> float:
        """Compute the entropy for the degree distribution of the graph.

        Returns:
            float: entropy of the degree distribution.
        """

        degrees = np.array([x for a, x in self.G.out_degree()])
        uniques, counts = np.unique(degrees, return_counts=True)

        # Frequencies are only determined to degrees existent in the network, degrees with a frequency of 0 are ignored
        freq = counts * (1 / self._n_nodes)
        self._raw_value = get_entropy(freq)

        self.h_max = math.log2(self._n_nodes)

        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the entropy of the degree distribution to the max theoretical entropy."""
        return self._raw_value / self.h_max

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the entropy of the degree distribution to the max theoretical entropy"""
        return self._raw_value / self.h_max


@return_scalar
@use_selfloops
@use_direction
class GiniIndex(_Property):
    """Gini Index.

    Measurement that reflects the inequiality of distribution of resources between entities. For networks, connections are
    considered as resources and each node is an individual entity. Therefore, we calculate how well (Gini Index = 0) or
    how unequal (Gini Index = 1) the distribution of links between nodes in the network is.
    It is calculated thorugh the area under the curve of The Lorenz Curve, which is drawn by the cummulative percentage of
    total connections which a certain fraction of nodes can have.

    Methods:
        compute: Compute the gini index for the graph.
        norm_biol: Normalize gini index to consider distribution of resources only between regulators.
        norm_network: Already normalized.
    """

    CLASS_NAME = "Gini Index"

    def __init__(self, G: nx.DiGraph):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        super().__init__(G)

    def compute(self) -> float:
        """Compute the gini index for the graph.

        Returns:
            float: gini index of the entire graph.
        """
        self.t = len(self.G.edges())
        if self.t == 0:
            raise EmptyGraphError(
                "There are no edges. Can not calculate Gini Index of a network with no edges."
            )

        b = [j for x, j in self.G.out_degree()]
        b.sort()
        area = 0

        for i in range(self._n_nodes):
            x = b[i] / self.t
            y = (self._n_nodes - (i + 1) + 0.5) / self._n_nodes
            area += x * y

        self._raw_value = 1 - (2 * area)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalization is a recalculation only between nodes with an out-degree higher than 0.
        Resources (connections) should not be distributed equally between all nodes in network, only between regulators"""
        n_parents = len(get_parent_nodes(self.G))
        b = [j for x, j in self.G.out_degree() if j != 0]
        b.sort()
        area = 0

        for i in range(n_parents):
            x = b[i] / self.t
            y = (n_parents - (i + 1) + 0.5) / n_parents
            area += x * y

        self._norm_biol = 1 - (2 * area)
        return self._norm_biol

    @check_raw_value
    def norm_network(self) -> float:
        """Already normalized."""
        return self._raw_value  # Already normalized [0,1]


@use_paths
@use_giant_component
@return_distribution
class BetweennessCentrality(_Property):
    """Betweenness Centrality.

    Betweenness centrality is a computed using the giant component of the graph.
    A high betweenness centrality indicates that a node is a bridge between different parts of the network.
    """

    CLASS_NAME = "Betweenness Centrality"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_paths = kwargs["net_shortest_paths"]
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the betweenness centrality for each node in the graph.

        Returns:
            np.array: Betweenness centrality for each node in the graph.
        """
        betweenness = self._shortest_paths.betweenness()
        self._raw_value = np.asarray(betweenness, dtype=float)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError(
            "Betweenness centrality cannot be normalized by biological properties."
        )

    @check_raw_value
    def norm_network(self) -> np.array:
        """Normalize betweenness centrality by the combinatory number of pairs of nodes excluding the node itself.

        Returns:
            np.array: Normalized betweenness centrality for each node in the graph.
        """
        scale_factor = 2 / ((self._n_nodes - 1) * (self._n_nodes - 2))
        return self._raw_value * scale_factor


@use_paths
@return_scalar
@use_giant_component
class GlobalEfficiency(_Property):
    """Global efficiency of a graph.

    The global efficiency of a graph is the average of the inverse of the shortest paths between all pairs of nodes.
    """

    CLASS_NAME = "Global Efficiency"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> float:
        """Compute the global efficiency of the graph.

        Returns:
            float: Global efficiency of the graph.
        """
        efficiency = Efficiency(self.G, self._shortest_distances)
        self._raw_value = efficiency.global_efficiency
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError("global efficiency cannot be normalized by biological properties.")

    @check_raw_value
    def norm_network(self) -> float:
        """Global efficiency is already normalized between 0 and 1."""
        # Already normalized
        return self._raw_value


@use_paths
@return_distribution
@use_giant_component
class Eccentricity(_Property):
    """Eccentricity.

    Eccentricity is a computed using the giant component of the graph.
    The eccentricity of a node is the largest shortest distance between that node and any other node in the graph.
    """

    CLASS_NAME = "Eccentricity"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> np.array:
        """Compute the eccentricity for each node in the graph.

        Returns:
            np.array: Eccentricity for each node in the graph.
        """
        self._raw_value = np.fromiter(self._shortest_distances.eccentricity().values(), dtype=float)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError("Eccentricity cannot be normalized by biological properties.")

    @check_raw_value
    def norm_network(self) -> np.array:
        """Normalize using the theoretical longest diameter in the graph using the giant component."""
        return self._raw_value / (self._n_nodes - 1)


@use_paths
@return_scalar
@use_giant_component
class Radius(_Property):
    """Radius.

    Radius is a computed using the giant component of the graph.
    The radius of a graph is the minimum eccentricity in the graph.
    """

    CLASS_NAME = "Radius"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> int:
        """Compute the radius of the graph.

        Returns:
            int: Radius of the graph.
        """
        self._raw_value = self._shortest_distances.radius
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError("Radius cannot be normalized by biological properties.")

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize using the theoretical longest diameter in the graph using the giant component."""
        return self._raw_value / (self._n_nodes - 1)


@use_paths
@return_scalar
@use_giant_component
class Center(_Property):
    """Center.

    Center is a computed using the giant component of the graph.
    The center of a graph is the set of nodes with eccentricity equal to the radius.
    This class returns the number of nodes in the center.
    """

    CLASS_NAME = "Center"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> int:
        """Compute the center for each node in the graph.

        Returns:
            int: Center for each node in the graph.
        """
        self._raw_value = len(self._shortest_distances.center)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError("Center cannot be normalized by biological properties.")

    @check_raw_value
    def norm_network(self) -> np.array:
        """Normalize using the number of nodes in the giant component."""
        return self._raw_value / self._n_nodes


@use_paths
@return_scalar
@use_giant_component
class Periphery(_Property):
    """Periphery.

    Periphery is a computed using the giant component of the graph.
    The periphery of a graph is the set of nodes with eccentricity equal to the diameter.
    This class returns the number of nodes in the periphery.
    """

    CLASS_NAME = "Periphery"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> int:
        """Compute the periphery for each node in the graph.

        Returns:
            int: Periphery for each node in the graph.
        """
        self._raw_value = len(self._shortest_distances.periphery)
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError("Periphery cannot be normalized by biological properties.")

    @check_raw_value
    def norm_network(self) -> int:
        """Normalize using the number of nodes in the giant component."""
        return self._raw_value / self._n_nodes


@use_paths
@return_scalar
@use_giant_component
class AverageLocalEfficiency(_Property):
    """Average Local Efficiency.

    Average Local Efficiency is a computed using the giant component of the graph.
    The average local efficiency of a graph is the average of the local efficiency of each node in the graph.
    The local efficiency of a node is the global efficiency computed over the neighborhood of the node.
    """

    CLASS_NAME = "Average Local Efficiency"

    def __init__(self, G: nx.DiGraph, **kwargs):
        """
        Args:
            G (nx.DiGraph): Graph.
        """
        if not G.number_of_edges():
            raise EmptyGraphError("Graph has no edges.")
        self._shortest_distances = kwargs["net_shortest_distances"]
        super().__init__(G)

    def compute(self) -> float:
        """Compute the average local efficiency for each node in the graph.

        Returns:
            float: Average local efficiency for each node in the graph.
        """
        efficiency = Efficiency(self.G, self._shortest_distances)
        self._raw_value = efficiency.local_efficiency
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> None:
        raise NormalizationError(
            "Average Local Efficiency cannot be normalized by biological properties."
        )

    @check_raw_value
    def norm_network(self) -> float:
        """Average Local Efficiency is already normalized between 0 and 1."""
        # Already normalized
        return self._raw_value

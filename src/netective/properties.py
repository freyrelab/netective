from __future__ import annotations

import math
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

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


def check_raw_value(func):
    """Decorator to check if raw value is None. If it is, raise an error."""

    def wrapper(self, *args):
        if self._raw_value is not None:
            return func(self, *args)
        else:
            raise ValueError("Raw value is None. Call compute() method first.")

    return wrapper


# Helper functions
def get_parent_nodes(G: nx.DiGraph):
    """Get the parent nodes of a graph."""
    return [i for i, k_out in G.out_degree() if k_out > 0]


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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        """Compute the maximum out-degree of the graph.

        Returns:
            int: Maximum out-degree of the graph.
        """
        self._raw_value = max(self.G.out_degree())
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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        """Compute the maximum in-degree of the graph.

        Returns:
            int: Maximum in-degree of the graph.
        """
        self._raw_value = max(self.G.in_degree())
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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = self.G.feedbacks3_count
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
        max_feedbacks3 = (
            self._n_nodes * (self._n_nodes - 1) * (self._n_nodes - 2)
        )  # TODO: verify
        return self._raw_value / max_feedbacks3


@return_scalar
@use_direction
class ComplexFeedForwardCircuits(_Property):
    """Number of complex feed-forward circuits.

    Methods:
        compute: Compute the number of complex feed-forward circuits.
        norm_biol: Normalize the number of complex feed-forward circuits to the number of parents.
        norm_network: Normalize the number of complex feed-forward circuits to the number of nodes.
    """

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = self.G.complex_feedforwards_count
        return self._raw_value

    def norm_biol(self) -> float:
        """Normalize the number of complex feed-forward circuits to the number of parents."""
        n_parents = len(get_parent_nodes(self.G))
        max_complex_ff = n_parents * (n_parents - 1)  # TODO: verify
        return self._raw_value / max_complex_ff

    def norm_network(self) -> float:
        """Normalize the number of complex feed-forward circuits to the number of nodes."""
        max_complex_ff = (
            self._n_nodes * (self._n_nodes - 1) * (self._n_nodes - 2)
        )  # TODO: verify
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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = self.G.giant_component_size
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
@use_direction
@use_giant_component
class Diameter(_Property):
    """Diameter of the graph.

    Methods:
        compute: Compute the diameter of the graph using the giant component.
        norm_biol: Normalize the diameter of the graph to the number of parents.
        norm_network: Normalize the diameter of the graph to the number of nodes.
    """

    def __init__(self, G: nx.DiGraph):
        G = self.G.giant_component
        super().__init__(G)

    def compute(self) -> int:
        self._raw_value = self.G.diameter()
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the diameter of the graph to the number of parents in the giant component.

        The maximum diameter of a graph with n nodes is n-1.
        When only considering the parents, the maximum diameter is n_parents - 1.
        Considering both, the maximum diameter is n_parents (every parent, then a leaf node).

        Note: the diameter of the giant component is not necessarily the same as the diameter of the graph.
        we are using the diameter of the giant component here given that the computation of the diameter is performed on the giant component.
        """
        n_parents = len(get_parent_nodes(self.G))
        return self._raw_value / n_parents

    @check_raw_value
    def norm_network(self) -> float:
        """Normalize the diameter of the graph to the number of nodes.

        The maximum diameter of a graph with n nodes is n-1.
        """
        return self._raw_value / (self._n_nodes - 1)


@use_paths
@return_scalar
@use_direction
@use_giant_component
class AverageShortestPathLength(_Property):
    """Average shortest path length of the graph.

    Methods:
        compute: Compute the average shortest path length of the graph using the giant component.
        norm_biol: Normalize the average shortest path length of the graph to the number of parents.
        norm_network: Normalize the average shortest path length of the graph to the number of nodes.
    """

    def __init__(self, G: nx.DiGraph):
        G = self.G.giant_component
        super().__init__(G)

    def compute(self) -> float:
        self._raw_value = self.G.average_shortest_path_length()
        return self._raw_value

    @check_raw_value
    def norm_biol(self) -> float:
        """Normalize the average shortest path length of the graph to the number of parents in the giant component."""
        n_parents = len(get_parent_nodes(self.G))
        return self._raw_value / n_parents

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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> float:
        self._raw_value = self.G.average_clustering_coefficient
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

    def __init__(self, G: nx.DiGraph):
        super().__init__(G)

    def compute(self) -> float:
        clustering_values = self.G.clustering_coefficients
        self._raw_value = np.fromiter(clustering_values.values(), dtype=float)
        return self._raw_value

    def norm_biol(self) -> float:
        """Coeffients are considered already normalized."""
        raise NotImplementedError  # TODO:!!! Change NormalizationError to NotImplementedError???

    def norm_network(self) -> float:
        """Coeffients are considered already normalized."""
        return (
            self._raw_value
        )  # TODO: o cambiar a NotImplementedError??? El usuario debería de decidir qué hacer en este caso, dejar nan o el valor crudo.

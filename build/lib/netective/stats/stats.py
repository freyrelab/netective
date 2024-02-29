from __future__ import annotations

import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations, product
from typing import Tuple

import numpy as np
from warnings import warn
from networkx import Graph, DiGraph

from netective.utils import remove_self_loops, parse_network

import logging
from netective.logging_info import get_logger, set_log_level
stats_logger = get_logger(__name__)

# TODO: SENT TO GLOBALS.PY
FONT_SIZE = 11
FACE_COLOR = "#F2F0F2"
FONT_COLOR = "#060307"
MAIN_PLOT_COLOR = "#031926"
MINOR_FONT_COLOR = "#504B51"

def _build_ax(ax=None, ylim=(0, 1), xlim=(0, 1), figsize=(2, 2)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Graph
    ax.set_facecolor(FACE_COLOR)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    return ax

def _universe(gold_standard_geneset: set, directed: bool, allow_self_loops: bool) -> set[tuple[str, str]]:
    """Returns the universe of potential edges between genes in the gold standard.

    The universe in compute following the rules:
    - If the inference is directed and self-loops are allowed, the universe is the cartesian product of the gold standard geneset.
    - If the inference is directed and self-loops are not allowed, the universe is the permutations of the gold standard geneset.
    - If the inference is undirected and self-loops are allowed, the universe is the combinations with replacement of the gold standard geneset.
    - If the inference is undirected and self-loops are not allowed, the universe is the combinations without replacement of the gold standard geneset.
    """
    if directed and allow_self_loops:
        return set(product(gold_standard_geneset, repeat=2))
    elif directed and not allow_self_loops:
        return set(permutations(gold_standard_geneset, 2))
    elif not directed and allow_self_loops:
        return set(combinations_with_replacement(gold_standard_geneset, 2))
    else:  # if not directed and not allow_self_loops:
        return set(combinations(gold_standard_geneset, 2))


def _anonymize_inference_edges(inference, gold_standard_geneset, greater_score_is_better, edge_to_id, directed):
        inference_edges = defaultdict(list)
        for u, v, data in inference.edges(data=True):
            # self-loops are already handled in __init__
            if u not in gold_standard_geneset or v not in gold_standard_geneset:
                continue    # we cannot assess what we don't know with this approach
            inference_edges[data["score"]].append((u, v))

        if not directed:
            inference_edges = {
                    score: {frozenset(edge) for edge in edges}
                    for score, edges in inference_edges.items()
                }
        
        # TODO: UX: The user may want to know the fraction of the inference used for the evaluation
        inference_edges = sorted(
            [
                [score, {edge_to_id[edge] for edge in edges}]
                for score, edges in inference_edges.items()
            ],
            reverse=greater_score_is_better,
        )
        return inference_edges

def _anonymize_edges(
    gold_standard: Graph | DiGraph,
    inference: Graph | DiGraph | list[Graph | DiGraph],
    greater_score_is_better: bool = True,
    allow_self_loops: bool = False,
    ) -> Tuple[set, list, int]:
    """Use the universe to anonymize the gold standard and the inference.
    The anonymization is done by mapping the edges to integers.

    Args:
    gold_standard (Graph | DiGraph): Gold standard network.
    inference (Graph | DiGraph | list[Graph | DiGraph]): Inference network. The edges can have a score as an attribute 'score'.
        If a list of networks is provided, a list of inference_edges is returned.
        The graphs must be of the same type (Graph or DiGraph) as the gold standard.
    greater_score_is_better (bool): Whether the inference score is better when it is higher or lower.
        If True, the higher the score, the better the inference.
        If False, the lower the score, the better the inference.
    allow_self_loops (bool): Whether self-loops are allowed or not.

    Returns:
    -------
    gold_standard_edges: set
        Set of edges in the gold standard.
    inference_edges: list
        List of lists containing the score and the set of edges for each score.
        The list is sorted by score following the greater_is_better rule.
        This is a list of lists when inference is a list of networks.
    size_universe: int
        Size of the universe of potential edges.
    """
    directed = gold_standard.is_directed()
    gold_standard_edges = set(gold_standard.edges(data=False))
    gold_standard_geneset = set(gold_standard.nodes(data=False))
    universe = _universe(gold_standard_geneset, directed, allow_self_loops)
    size_universe = len(universe)

    

    if not directed:
        # If the inference is not directed, we use frozensets to ignore gene order in the edges
        gold_standard_edges = {frozenset(edge) for edge in gold_standard_edges}
        universe = {frozenset(edge) for edge in universe}
        # The size of the universe should not change, repetition are considered in __universe

    # Universe is used as reference
    edge_to_id = {edge: i for i, edge in enumerate(universe)}
    gold_standard_edges = {edge_to_id[edge] for edge in gold_standard_edges}

    if isinstance(inference, list):
        inference_edges = [_anonymize_inference_edges(
            inf, gold_standard_geneset, greater_score_is_better, edge_to_id, directed
            ) for inf in inference]
    else: # inference is a single network
        inference_edges = _anonymize_inference_edges(
            inference, gold_standard_geneset, greater_score_is_better, edge_to_id, directed
            )

    # erase universe, mapping and gold standard geneset
    del universe
    del edge_to_id  # TODO: UX: user may want to keep this mapping
    del gold_standard_geneset
    # call garbage collector
    gc.collect()

    return gold_standard_edges, inference_edges, size_universe


def benchmarking(
    networks: dict | str,
    gold_standard: Graph | DiGraph | str,
    directed: bool = True,
    greater_score_is_better: bool = True,
    allow_self_loops: bool = False,
    cutoff: float | False = False,
    return_auc_dicts: bool = False,
    comments : str = '#',
    delimiter : str = '\t',
    score : bool = True,
    verbose: str = None,
) -> None:
    """Perform a statistical analysis of the inference networks.
    
    Args:
        networks (dict | str): Dictionary containing the inference networks or path to the directory containing the inference networks.
            If a dictionary is provided, the keys are the network names and the values are the networks.
            If a path is provided, the networks are loaded from the directory.
        gold_standard (Graph | DiGraph | str): Networkx object or name or path to the gold standard network.
            If gold_standard is part of the networks, it is removed from the networks.
            If networks is a path to a directory, gold_standard must be a path to a file.
        directed (bool): Whether the gold standard and the inference networks are directed or not.
        greater_score_is_better (bool): Whether the inference score is better when it is higher or lower.
            If True, the higher the score, the better the inference.
            If False, the lower the score, the better the inference.
        allow_self_loops (bool): Whether self-loops are allowed or not.
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If False, the evaluation metrics are computed for every score in the inference.
        return_auc_dicts (bool): Whether to return the AUC values for every inference in the benchmark.
            If False, the figure axis are returned.
        comments (str): Character used to indicate comments in the network files.
        delimiter (str): Character used to separate the columns in the network files.
        score (bool): Whether the inference networks have a score attribute or not.
            If score is False, the ranking of the edges is used as score (e.g., the first edge has a score of 0, the second a score of 1, etc.).
        verbose (str): Level of verbosity.
            If None, the verbosity level is set to WARNING. Options are: DEBUG, INFO, WARNING, ERROR, CRITICAL.

        
    Returns:
        If return_auc_dicts is True, a tuple containing the AUPR and AUROC dicts with values for every inference in the benchmark.
        If return_auc_dicts is False, a tuple containing the figure axis for the AUPR and AUROC plots.
    """

    if verbose is not None:
        current_level = stats_logger.getEffectiveLevel()
        set_log_level(stats_logger, verbose)

    if isinstance(networks, str):

        # valid networks dir and gold standard path
        if not os.path.isdir(networks):
            raise ValueError("networks must be a dictionary or a path to a directory.")
        if not os.path.isfile(gold_standard):
            raise ValueError("gold_standard must be a path to a file when networks is a path to a directory.")

        # load networks using the filename as key
        networks = {
            os.path.splitext(network)[0]: parse_network(os.path.join(networks, network), comments=comments, delimiter=delimiter, directed=directed, score=score, use_position_as_score=True if not score else False)
            for network in os.listdir(networks) if network != gold_standard
            }
        
        gold_standard = parse_network(gold_standard, comments=comments, delimiter=delimiter, directed=directed, score=False, use_position_as_score=False)
    
    elif isinstance(networks, dict):
        if gold_standard in networks:
            gold_standard = networks.pop(gold_standard)
        else:
            if not isinstance(gold_standard, (Graph, DiGraph)):
                raise TypeError("gold_standard must be a networkx.Graph or networkx.DiGraph.")
            gold_standard = gold_standard.copy()

    else:
        raise TypeError("networks must be a dictionary or a path to a directory.")
    
    benchmark = Benchmark(
        gold_standard=gold_standard,
        inferences=networks,
        greater_score_is_better=greater_score_is_better,
        allow_self_loops=allow_self_loops,
        cutoff=cutoff,
    )


    if return_auc_dicts:
        return_set = benchmark.aupr(), benchmark.auroc()
    
    else:
        fig_aupr = benchmark.plot_aupr()
        fig_pr_curves = benchmark.plot_precision_recall_curves()
        fig_auroc = benchmark.plot_auroc()
        fig_roc_curves = benchmark.plot_roc_curves()
        return_set = fig_aupr, fig_pr_curves, fig_auroc, fig_roc_curves

    if verbose is not None:
        set_log_level(stats_logger, current_level)

    return return_set


class Benchmark:
    def __init__(
        self,
        gold_standard: Graph | DiGraph,
        inferences: dict[str, Graph | DiGraph],
        greater_score_is_better: bool = True,
        allow_self_loops: bool = False,
        cutoff: float | False = False,
    ):
        """
        """
        self.__repr_str = f"Benchmark({gold_standard}, {inferences}, greater_is_better={greater_score_is_better}, allow_self_loops={allow_self_loops}, cutoff={cutoff})"
        

        # remove self-loops if allow_self_loops is False
        if not allow_self_loops:
            gold_standard = remove_self_loops(gold_standard)
            inferences = {net_id: remove_self_loops(inf) for net_id, inf in inferences.items()}

        gold_standard_edges, inference_edges, size_universe = _anonymize_edges(
            gold_standard,
            list(inferences.values()),
            greater_score_is_better,
            allow_self_loops,
        )


        self.__nis_instances = {
            net_id: LinkEval(
                cutoff=cutoff,
                gold_standard_edges=gold_standard_edges,
                inference_edges=inf,
                size_universe=size_universe,
            )
            for net_id, inf in zip(inferences.keys(), inference_edges)
        }

    @property
    def nis_instances(self) -> dict[str, LinkEval]:
        return self.__nis_instances
    
    def __getitem__(self, key: str) -> LinkEval:
        return self.__nis_instances[key]
    
    def __repr__(self) -> str:
        return self.__repr_str
    
    def __str__(self) -> str:
        return f"Benchmark({len(self.nis_instances)} instances)"
    
    def __eq__(self, other: Benchmark) -> bool:
        raise NotImplementedError
    
    def plot_roc_curves(self, ax=None, **kwargs):
        """Plots the ROC curves for every inference in the benchmark.

        Args:
        ax (matplotlib.axes.Axes): Axes object to plot the curve.
            If None, a new figure and axes are created.
        **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
        -------
        ax: matplotlib.axes.Axes
            Axes object containing the plot.
        """
        ax = _build_ax(ax, figsize=(3,3))
        best_name, _ = self.best_auroc()
        for net_id, nis in self.nis_instances.items():
            nis.plot_roc_curve(ax=ax, label=net_id if net_id==best_name else None, color='r' if net_id==best_name else MAIN_PLOT_COLOR, alpha=0, title=False, **kwargs)
        ax.legend(loc=3)
        return ax
    
    def plot_precision_recall_curves(self, ax=None, **kwargs):
        """Plots the precision-recall curves for every inference in the benchmark.

        Args:
        ax (matplotlib.axes.Axes): Axes object to plot the curve.
            If None, a new figure and axes are created.
        **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
        -------
        ax: matplotlib.axes.Axes
            Axes object containing the plot.
        """
        ax = _build_ax(ax, figsize=(3,3))
        best_name, _ = self.best_aupr()
        for net_id, nis in self.nis_instances.items():
            nis.plot_precision_recall_curve(ax=ax, label=net_id if net_id==best_name else None, color='r' if net_id==best_name else MAIN_PLOT_COLOR, alpha=0, title=False, **kwargs)
        ax.legend(loc=3)
        return ax
    
    def optimal_cutoffs(self) -> dict[str, float]:
        """Computes the optimal cutoff for every inference in the benchmark.

        Returns:
        -------
        optimal_cutoff: dict[str, float]
            Dictionary containing the optimal cutoff for every inference in the benchmark.
        """
        return {net_id: nis.optimal_cutoff() for net_id, nis in self.nis_instances.items()}
    
    def plot_optimal_cutoffs(self, ax=None, **kwargs):
        """Plots the optimal cutoff values for every inference in the benchmark."""
        ax = _build_ax(ax, xlim=None, ylim=None)
        optimal_cutoffs = self.optimal_cutoffs()
        ax.barh(list(optimal_cutoffs.keys()), list(optimal_cutoffs.values()), color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("Optimal cutoff", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("Cutoff", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_aupr(self, ax=None, **kwargs):
        """Plots the AUPR values for every inference in the benchmark."""
        ax = _build_ax(ax, xlim=[0,1.01], ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.area_under_precision_recall_curve() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("AUPR", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("AUPR", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_auroc(self, ax=None, **kwargs):
        """Plots the AUROC values for every inference in the benchmark."""
        ax = _build_ax(ax, xlim=[0,1.01], ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.area_under_roc_curve() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("AUROC", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("AUROC", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_f1_score(self, ax=None, **kwargs):
        """Plots the F1 score values for every inference in the benchmark."""
        ax = _build_ax(ax, xlim=[0,1.01], ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.f1_score() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("F1 score", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("F1 score", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_mcc(self, ax=None, **kwargs):
        """Plots the MCC values for every inference in the benchmark."""
        ax = _build_ax(ax, xlim=[-1.01,1.01], ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.mcc() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("MCC", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("MCC", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def best_aupr(self) -> tuple(str, float):
        """Returns the inference with the best AUPR and its value."""
        return max([(net_id, nis.area_under_precision_recall_curve()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def best_auroc(self) -> tuple(str, float):
        """Returns the inference with the best AUROC and its value."""
        return max([(net_id, nis.area_under_roc_curve()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])

    def best_f1_score(self) -> tuple(str, float):
        """Returns the inference with the best F1 score and its value."""
        return max([(net_id, nis.f1_score()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def best_accuracy(self) -> tuple(str, float):
        """Returns the inference with the best accuracy and its value."""
        return max([(net_id, nis.accuracy()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def best_recall(self) -> tuple(str, float):
        """Returns the inference with the best recall and its value."""
        return max([(net_id, nis.recall()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def best_precision(self) -> tuple(str, float):
        """Returns the inference with the best precision and its value."""
        return max([(net_id, nis.precision()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def best_mcc(self) -> tuple(str, float):
        """Returns the inference with the best MCC and its value."""
        return max([(net_id, nis.mcc()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    def aupr(self) -> dict[str, float]:
        """Returns the AUPR values for every inference in the benchmark."""
        return {net_id: nis.area_under_precision_recall_curve() for net_id, nis in self.nis_instances.items()}
    
    def auroc(self) -> dict[str, float]:
        """Returns the AUROC values for every inference in the benchmark."""
        return {net_id: nis.area_under_roc_curve() for net_id, nis in self.nis_instances.items()}
    
    def coordinates(self, cutoff=None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns the coordinates for the precision, recall and FPR for every inference in the benchmark."""
        return {net_id: nis.coordinates(cutoff) for net_id, nis in self.nis_instances.items()}


class LinkEval:
    def __init__(
        self,
        gold_standard: Graph | DiGraph = None,
        inference: Graph | DiGraph = None,
        greater_score_is_better: bool = None,
        allow_self_loops: bool = None,
        cutoff: float | False = False,
        gold_standard_edges: set = None,
        inference_edges: list = None,
        size_universe: int = None,
    ):
        """
        Class for evaluating binary classification results.
        Optimized for network inference.

        Args:
        gold_standard (Graph | DiGraph): Gold standard network.
        inference (Graph | DiGraph): Inference network. The edges can have a score as an attribute 'score'.
        greater_score_is_better (bool): Whether the inference score is better when it is higher or lower.
            If True, the higher the score, the better the inference.
            If False, the lower the score, the better the inference.
        allow_self_loops (bool): Whether self-loops are allowed or not.
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If False, the evaluation metrics are computed for every score in the inference.
        gold_standard_edges (set): Set of edges in the gold standard.
            If None, the gold standard edges are computed from the gold standard network.
            Use only if the gold standard edges are already known.
        inference_edges (list): List of lists containing the score and the set of edges for each score.
            The list is sorted by score following the greater_is_better rule.
            Use only if the inference edges are already known.
        size_universe (int): Size of the universe of potential edges.
            If None, the size of the universe is computed from the gold standard network.
            Use only if the size of the universe is already known.

        Notes:
        Networks types must be the same (Graph or DiGraph).
        Node IDs for gold standard and inference must be comparable.
        Nodes in the inference not present in the gold standard will be ignored as gold standard may not be complete.
        """

        self.__gold_standard = gold_standard
        self.__inference = inference
        self.__greater_is_better = greater_score_is_better
        self.__allow_self_loops = allow_self_loops

        if all([x is not None for x in [gold_standard_edges, inference_edges, size_universe]]):
            self.__gold_standard_edges = gold_standard_edges
            self.__inference_edges = inference_edges
            self.__size_universe = size_universe
            self.__repr = f"LinkEval(gold_standard_edges={len(self.gold_standard_edges)}, inference_edges={len(self.inference_edges)}, size_universe={self.size_universe})"

        else:
            self.__directed = self.gold_standard.is_directed()
            
            if gold_standard is None or inference is None:
                raise ValueError("Gold standard and inference must be provided if gold_standard_edges, inference_edges and size_universe are not provided.")

            self.__validate_networks()

            if not self.allow_self_loops:
                self.__gold_standard = remove_self_loops(self.gold_standard)
                self.__inference = remove_self_loops(self.inference)
                
            self.__gold_standard_edges, self.__inference_edges, self.__size_universe = _anonymize_edges(
                self.gold_standard,
                self.inference,
                self.greater_is_better,
                self.allow_self_loops,
            )

            self.__repr = f"LinkEval(gold_standard={self.__gold_standard}, inference={self.__inference}, greater_is_better={self.__greater_is_better}, directed={self.__directed})"

        
        self.__size_gold_standard = len(self.__gold_standard_edges)
        self.__size_negatives = self.__size_universe - self.__size_gold_standard
        # At the last step, every edge is considered as a positive by inference
        self.__precision_baseline = (self.size_gold_standard / self.size_universe)  # (GS/(GS + (Universe-GS)))
        if not cutoff:
            self.__cutoff = min([score for score, _ in self.inference_edges]) if self.greater_is_better else max([score for score, _ in self.inference_edges])

        # Define evaluation
        # Used as flags to know if the curves data points have been computed
        self.__fpr_dist = None
        self.__precision_dist = None
        self.__sensitivity_dist = None
        self.__f1_score_dist = None

    def __repr__(self) -> str:
        return self.__repr

    def __str__(self) -> str:
        return self.__repr

    def __eq__(self, other: LinkEval) -> bool:
        raise NotImplementedError

    @property
    def cutoff(self) -> float | None:
        """Return the less restrictive score used to compute the evaluation metrics."""
        return self.__cutoff

    @cutoff.setter
    def cutoff(self, cutoff: float | None):
        self.__cutoff = cutoff
        # Reset the cache
        self.__precision_dist = None
        self.__sensitivity_dist = None
        self.__fpr_dist = None
        self.__f1_score_dist = None

    @property
    def precision_dist(self) -> np.ndarray:
        return self.__precision_dist

    @property
    def sensitivity_dist(self) -> np.ndarray:
        return self.__sensitivity_dist

    @property
    def fpr_dist(self) -> np.ndarray:
        return self.__fpr_dist

    @property
    def precision_baseline(self) -> float:
        return self.__precision_baseline

    @property
    def size_negatives(self) -> int:
        return self.__size_negatives

    @property
    def size_gold_standard(self) -> int:
        return self.__size_gold_standard

    @property
    def gold_standard_edges(self) -> set:
        return self.__gold_standard_edges

    @property
    def inference_edges(self) -> dict[float, set]:
        return self.__inference_edges

    @property
    def size_universe(self) -> int:
        return self.__size_universe

    @property
    def gold_standard(self) -> str:
        return self.__gold_standard

    @property
    def inference(self) -> str:
        return self.__inference

    @property
    def greater_is_better(self) -> bool | None:
        return self.__greater_is_better

    @property
    def directed(self) -> bool:
        return self.__directed

    @property
    def allow_self_loops(self) -> bool:
        return self.__allow_self_loops
    
    def __validate_networks(self):
        for name, network in [("Gold standard", self.gold_standard), ("Inference", self.inference)]:
            if not isinstance(network, (Graph, DiGraph)):
                raise TypeError(f"{name} must be a networkx.Graph or networkx.DiGraph.")

        if self.gold_standard.is_directed() != self.inference.is_directed():
            raise ValueError("Gold standard and inference must be both directed or undirected.")

        if not all("score" in data for _, _, data in self.inference.edges(data=True)):
            raise ValueError("Inference edges must have a score attribute.")

        if any("score" in data for _, _, data in self.gold_standard.edges(data=True)):
            warn("The gold standard edges have a score attribute. It might be a prediction.")


    def __compute_roc_pr_datapoints(self, cutoff=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the false positive rate, sensitivity and precision for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None (default) the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.


        Returns:
        precision: np.ndarray
            Precision values for the given cutoff.
        sensitivity: np.ndarray
            Sensitivity values for the given cutoff.
        fpr: np.ndarray
            False positive rate values for the given cutoff.

        Notes:
        It caches the values for the given cutoff if the cutoff is the same as the one provided in the initialization.
        The inference edges must be sorted by score following the greater_is_better rule.
        It returns the cache values if the cutoff is the same as the one provided in the initialization or if cutoff is None.
        """
        cutoff = self.__validate_cutoff(cutoff)
        cache = True if cutoff == self.cutoff else False
        # TODO:!!! optimization: use a subset of the computed datapoints.

        if (
            cache
            and self.precision_dist is not None
            and self.sensitivity_dist is not None
            and self.fpr_dist is not None
        ):
            stats_logger.warning("The evaluation metrics have already been computed for this cutoff. Returning cached values.")
            return self.precision_dist, self.sensitivity_dist, self.fpr_dist

        if cutoff == self.cutoff:
            inference_edges = [edges for _, edges in self.inference_edges]
        else:
            if self.greater_is_better:
                inference_edges = [
                    edges for score, edges in self.inference_edges if score >= cutoff
                ]
            else:
                inference_edges = [
                    edges for score, edges in self.inference_edges if score <= cutoff
                ]

        num_points = len(inference_edges) + 2
        # Initialize arrays to store the evaluation metrics coordinates
        fpr_dist = np.empty(num_points)
        precision_dist = np.empty(num_points)
        sensitivity_dist = np.empty(num_points)  # same as recall and TPR

        # Start-values
        fpr_dist[0] = 0
        precision_dist[0] = 0
        sensitivity_dist[0] = 0

        predicted_positives = set()
        for i, edges in enumerate(inference_edges, 1):
            predicted_positives.update(edges)
            true_positives = len(self.gold_standard_edges & predicted_positives)
            false_positives = len(predicted_positives - self.gold_standard_edges)
            # remplace the corresponding values in the arrays
            fpr_dist[i] = false_positives / self.size_negatives
            sensitivity_dist[i] = true_positives / self.size_gold_standard
            precision_dist[i] = true_positives / (true_positives + false_positives)

        # End-values
        precision_dist[-1] = self.precision_baseline
        sensitivity_dist[-1] = 1  # no FN (no negatives)
        fpr_dist[-1] = 1  # no TN (no negatives)

        # The first precision values must equal the second one
        precision_dist[0] = precision_dist[1]

        if cache:
            # enters only when the three arrays are None
            self.__precision_dist = precision_dist
            self.__sensitivity_dist = sensitivity_dist
            self.__fpr_dist = fpr_dist

        return precision_dist, sensitivity_dist, fpr_dist

    def __compute_auc(self, x, y) -> float:
        """Computes the area under the curve using the trapezoidal rule."""
        return np.trapz(x=x, y=y, dx=0.05)

    def area_under_precision_recall_curve(self, cutoff=None) -> float:
        """Computes the area under the precision-recall curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used. Use False to avoid using a cutoff.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.

        Returns:
        -------
        area: float
            Area under the precision-recall curve.
        """
        precision, sensitivity, _ = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        stats_logger.debug(f'precision: {precision}, sensitivity: {sensitivity}')
        return self.__compute_auc(x=sensitivity, y=precision)

    def area_under_roc_curve(self, cutoff=None) -> float:
        """Computes the area under the ROC curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used. Use False to avoid using a cutoff.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.

        Returns:
        -------
        area: float
            Area under the ROC curve.
        """
        _, sensitivity, fpr = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        return self.__compute_auc(x=fpr, y=sensitivity)


    def __plot_curve(self, x, y, xlabel, ylabel, title="AUC", ax=None, color=MAIN_PLOT_COLOR, alpha = 0.8, **kwargs):
        ax = _build_ax(ax, xlim=(0, 1.02), ylim=(0, 1.02))
        ax.fill_between(x, y, color=color, alpha=alpha)
        ax.plot(x, y, color=color, alpha=1, **kwargs)
        # Add titles
        ax.set_title(title, loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_xlabel(xlabel, size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_ylabel(ylabel, size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def __validate_cutoff(self, cutoff):

        if cutoff is None or cutoff==self.cutoff:
            return self.cutoff

        if self.greater_is_better:
            if cutoff is not None and cutoff < self.cutoff:
                raise ValueError(
                    f"The cutoff must be greater than the one provided in the initialization ({self.cutoff})."
                )
        else:
            if cutoff is not None and cutoff > self.cutoff:
                raise ValueError(
                    f"The cutoff must be lower or equal than the one provided in the initialization ({self.cutoff})."
                )
        return cutoff

    def plot_precision_recall_curve(self, cutoff=None, ax=None, title=True, **kwargs):
        """Plots the precision-recall curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the evaluation metrics are computed for every score in the inference.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.
        ax (matplotlib.axes.Axes): Axes object to plot the curve.
            If None, a new figure and axes are created.
        **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
        -------
        ax: matplotlib.axes.Axes
            Axes object containing the plot.
        """
        precision, sensitivity, _ = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        ax = self.__plot_curve(
            sensitivity,
            precision,
            xlabel="Recall",
            ylabel="Precision",
            title=f"AUC-PR = {self.__compute_auc(x=sensitivity, y=precision):.3f}" if title else None,
            ax=ax,
            **kwargs,
        )
        return ax

    def plot_roc_curve(self, cutoff=None, ax=None, title=True, **kwargs):
        """Plots the ROC curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the evaluation metrics are computed for every score in the inference.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.
        ax (matplotlib.axes.Axes): Axes object to plot the curve.
            If None, a new figure and axes are created.
        **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
        -------
        ax: matplotlib.axes.Axes
            Axes object containing the plot.
        """
        _, sensitivity, fpr = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        ax = self.__plot_curve(
            self.fpr_dist,
            self.sensitivity_dist,
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"AUC-ROC = {self.__compute_auc(x=fpr, y=sensitivity):.3f}" if title else None,
            ax=ax,
            **kwargs,
        )
        return ax

    def __filtered_inference_edges(self, cutoff=None) -> set:
        """Returns the set of edges predicted by the inference for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, raises an error.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization.

        Returns:
        -------
        predicted_edges: set
            Set of edges predicted by the inference for the given cutoff.
        """
        if cutoff is None:
            raise ValueError("A cutoff must be provided.")
        if self.greater_is_better:
            return set().union(*[edges for score, edges in self.inference_edges if score >= cutoff])
        else:
            return set().union(*[edges for score, edges in self.inference_edges if score <= cutoff])

    # TODO: Optimization: Create a cache to keep {(cutoff, metric): value)}
    def recall(self, cutoff:None|float=None) -> float:
        """Computes the recall for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization.

        Returns:
        recall: float
            Recall for the given cutoff.
        """
        cutoff = self.__validate_cutoff(cutoff)

        if cutoff == self.cutoff and self.__sensitivity_dist is not None:
            sensitivity = self.__sensitivity_dist[-2]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            sensitivity = true_positives / self.size_gold_standard
        return sensitivity

    def precision(self, cutoff:None|float=None) -> float:
        """Computes the precision for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization.
        
        Returns:
        precision: float
            Precision for the given cutoff.
        """
        cutoff = self.__validate_cutoff(cutoff)
        
        if cutoff == self.cutoff and self.__precision_dist is not None:
            precision = self.__precision_dist[-2]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            precision = true_positives / (true_positives + false_positives)
        return precision
    
    def fpr(self, cutoff:None|float) -> float:
        """Computes the false positive rate for a given cutoff.
        
        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization.

        Returns:
        fpr: float
            False positive rate for the given cutoff.
        """
        cutoff = self.__validate_cutoff(cutoff)

        if cutoff == self.cutoff and self.__fpr_dist is not None:
            fpr = self.__fpr_dist[-2]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            fpr = false_positives / self.size_negatives
        return fpr
    
    def accuracy(self, cutoff:None|float=None) -> float:
        """Computes the accuracy for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization.

        Returns:
        accuracy: float
            Accuracy for the given cutoff.
        """
        cutoff = self.__validate_cutoff(cutoff)
        if cutoff == self.cutoff:
            predicted_edges = set().union(*[edges for _, edges in self.inference_edges])
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
        true_positives = len(self.gold_standard_edges & predicted_edges)
        false_positives = len(predicted_edges - self.gold_standard_edges)
        false_negatives = self.size_gold_standard - true_positives
        true_negatives = self.size_negatives - false_positives


        return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    

    def f1_score(self, cutoff:None|float=None) -> float:
        """Computes the F1 score for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.

        Returns:
        f1_score: float
            F1 score for the given cutoff.
        """
        if precision+recall == 0:
            return 0
        cutoff = self.__validate_cutoff(cutoff)
        precision = self.precision(cutoff=cutoff)
        recall = self.recall(cutoff=cutoff)
        return 2 * (precision * recall) / (precision + recall)
    
    def mcc(self, cutoff:None|float=None) -> float:
        """Computes the Matthews correlation coefficient for a given cutoff.

        The mcc ranges from -1 to 1, where 1 is a perfect prediction, 0 is a random prediction and -1 is a perfectly wrong prediction.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used.
            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided in the initialization or the cutoff value set.

        Returns:
        mcc: float
            Matthews correlation coefficient for the given cutoff.
        """
        cutoff = self.__validate_cutoff(cutoff)
        predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
        true_positives = len(self.gold_standard_edges & predicted_edges)
        false_positives = len(predicted_edges - self.gold_standard_edges)
        false_negatives = self.size_gold_standard - true_positives
        true_negatives = self.size_negatives - false_positives

        return (true_positives * true_negatives - false_positives * false_negatives) / np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
    
    def __compute_f1_score_dist(self) -> np.ndarray:
        """Computes the F1 score for every score in the inference.

        Returns:
        f1_score_dist: dict[float, float]
            Dictionary containing the F1 score for every score in the inference.
        """
        f1_score_dist = {}
        for score, _ in self.inference_edges:
            f1_score_dist[score] = self.f1_score(cutoff=score)
        self.__f1_score_dist = f1_score_dist
        return f1_score_dist
    
    def optimal_cutoff(self) -> float:
        """Computes the optimal cutoff for the inference, that required to maximize the F1 score.

        Returns:
        optimal_cutoff: float
            Optimal cutoff for the inference.
        """
        self.__f1_score_dist = self.__f1_score_dist if self.__f1_score_dist is not None else self.__compute_f1_score_dist()
        return max(self.__f1_score_dist, key=self.__f1_score_dist.get)
    
    def optimal_cutoff_plot(self, ax=None, x_log=False):
        """Plots the precision and the recall values for every score in the inference, showing the optimal cutoff.

        Args:
        ax (matplotlib.axes.Axes): Axes object to plot the curve.
            If None, a new figure and axes are created.
        **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
        -------
        ax: matplotlib.axes.Axes
            Axes object containing the plot.
        """
        self.__f1_score_dist = self.__f1_score_dist if self.__f1_score_dist is not None else self.__compute_f1_score_dist()
        optimal_cutoff = self.optimal_cutoff()
        precision_dist, sensitivity_dist, _ = self.__compute_roc_pr_datapoints() # use the default cutoff to get the cached values
        ax = _build_ax(ax, xlim=(min(self.__f1_score_dist), max(self.__f1_score_dist)), figsize=(4, 2))
        ax.plot(self.__f1_score_dist.keys(), precision_dist[1:-1], label="Precision", color="#DC3220")
        ax.plot(self.__f1_score_dist.keys(), sensitivity_dist[1:-1], label="Recall", color="#005AB5")
        ax.vlines(optimal_cutoff, ymin=0, ymax=1, color="k", linestyles='dashed', label="Optimal cutoff")
        ax.set_title(f"Optimal cutoff = {optimal_cutoff}", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_xlabel("Score", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_ylabel("Precision and recall", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.legend(loc=0)
        if x_log:
            ax.set_xscale("log")
        stats_logger.debug(self.__f1_score_dist.values(), precision_dist[1:-1], sensitivity_dist[1:-1])
        return ax
    

    def coordinates(self, cutoff=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the coordinates for the precision, recall and FPR for every score in the inference.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the score provided in the initialization is used.

        Returns:
        -------
        precision: np.ndarray
            Precision values for every score in the inference.
        sensitivity: np.ndarray
            Sensitivity values for every score in the inference.
        fpr: np.ndarray
            False positive rate values for every score in the inference.
        """
        return self.__compute_roc_pr_datapoints(cutoff=cutoff)

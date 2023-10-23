from __future__ import annotations

import gc
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations, product
from typing import Tuple

import numpy as np
from warnings import warn
from networkx import Graph, DiGraph

from netective.utils import remove_self_loops

# SENT TO GLOBALS.PY
FONT_SIZE = 11
FACE_COLOR = "#F2F0F2"
FONT_COLOR = "#060307"
MAIN_PLOT_COLOR = "#031926"
MINOR_FONT_COLOR = "#504B51"


class NetworkInferenceStats:
    def __init__(
        self,
        gold_standard: Graph | DiGraph,
        inference: Graph | DiGraph,
        greater_score_is_better: bool = True,
        allow_self_loops: bool = False,
        cutoff: float | False = False,
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

        Notes:
        Networks types must be the same (Graph or DiGraph).
        Node IDs for gold standard and inference must be comparable.
        Nodes in the inference not present in the gold standard will be ignored as gold standard may not be complete.
        """

        self.__gold_standard = gold_standard
        self.__inference = inference
        self.__greater_is_better = greater_score_is_better
        self.__allow_self_loops = allow_self_loops
        self.__cutoff = cutoff

        # Validate the networks
        for name, network in [("Gold standard", self.gold_standard), ("Inference", self.inference)]:
            if not isinstance(network, (Graph, DiGraph)):
                raise TypeError(f"{name} must be a networkx.Graph or networkx.DiGraph.")

        if self.gold_standard.is_directed() != self.inference.is_directed():
            raise ValueError("Gold standard and inference must be both directed or undirected.")

        if not all("score" in data for _, _, data in self.inference.edges(data=True)):
            raise ValueError("Inference edges must have a score attribute.")

        if any("score" in data for _, _, data in self.gold_standard.edges(data=True)):
            warn("The gold standard edges have a score attribute. It might be a prediction.")

        self.__directed = self.gold_standard.is_directed()
        if not self.allow_self_loops:
            self.__gold_standard = remove_self_loops(self.gold_standard)
            self.__inference = remove_self_loops(self.inference)

        # Define evaluation
        # Used as flags to know if the curves data points have been computed
        self.__fpr = None
        self.__precision = None
        self.__sensitivity = None

        (
            self.__gold_standard_edges,
            self.__inference_edges,
            self.__size_universe,
        ) = self.__anonymize_edges()
        self.__size_gold_standard = len(self.__gold_standard_edges)
        self.__size_negatives = self.__size_universe - self.__size_gold_standard

    def __repr__(self) -> str:
        return f"BinClassEval(gold_standard={self.gold_standard}, inference={self.inference}, greater_is_better={self.greater_is_better}, directed={self.directed})"

    def __str__(self) -> str:
        if self.__num_columns == 3:
            return f'NetworkInferenceStats to evaluate the inference {self.inference} against {self.gold_standard} {"" if self.directed else "not "}\
                considering direction and {"higher" if self.greater_is_better else "lower"} scores are better.'
        else:  # self._num_columns == 2
            return f'NetworkInferenceStats to evaluate the inference {self.inference} against {self.gold_standard} {"" if self.directed else "not "}\
                considering direction.'

    def __eq__(self, other: NetworkInferenceStats) -> bool:
        raise NotImplementedError

    @property
    def cutoff(self) -> float | None:
        return self.__cutoff

    @cutoff.setter
    def cutoff(self, cutoff: float | None):
        self.__cutoff = cutoff
        # Reset the cache
        self.__precision = None
        self.__sensitivity = None
        self.__fpr = None

    @property
    def precision(self) -> np.ndarray:
        return self.__precision

    @property
    def sensitivity(self) -> np.ndarray:
        return self.__sensitivity

    @property
    def fpr(self) -> np.ndarray:
        return self.__fpr

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

    def __universe(self, gold_standard_geneset: set) -> set[tuple[str, str]]:
        """Returns the universe of potential edges between genes in the gold standard.

        The universe in compute following the rules:
        - If the inference is directed and self-loops are allowed, the universe is the cartesian product of the gold standard geneset.
        - If the inference is directed and self-loops are not allowed, the universe is the permutations of the gold standard geneset.
        - If the inference is undirected and self-loops are allowed, the universe is the combinations with replacement of the gold standard geneset.
        - If the inference is undirected and self-loops are not allowed, the universe is the combinations without replacement of the gold standard geneset.
        """
        if self.directed and self.allow_self_loops:
            return set(product(gold_standard_geneset, repeat=2))
        elif self.directed and not self.allow_self_loops:
            return set(permutations(gold_standard_geneset, 2))
        elif not self.directed and self.allow_self_loops:
            return set(combinations_with_replacement(gold_standard_geneset, 2))
        else:  # if not self.directed and not self.allow_self_loops:
            return set(combinations(gold_standard_geneset, 2))

    def __anonymize_edges(self) -> Tuple[set, list, int]:
        """Use the universe to anonymize the gold standard and the inference.
        The anonymization is done by mapping the edges to integers.

        Returns:
        -------
        gold_standard_edges: set
            Set of edges in the gold standard.
        inference_edges: list
            List of lists containing the score and the set of edges for each score.
            The list is sorted by score following the greater_is_better rule.
        size_universe: int
            Size of the universe of potential edges.
        """
        gold_standard_edges = set(self.gold_standard.edges(data=False))
        gold_standard_geneset = set(self.gold_standard.nodes(data=False))
        inference_edges = defaultdict(list)
        for u, v, data in self.inference.edges(data=True):
            # self-loops are already handled in __init__
            if u not in gold_standard_geneset or v not in gold_standard_geneset:
                continue
            inference_edges[data["score"]].append((u, v))

        universe = self.__universe(gold_standard_geneset)
        size_universe = len(universe)

        if not self.directed:
            # If the inference is not directed, we use frozensets to ignore gene order in the edges
            gold_standard_edges = {frozenset(edge) for edge in gold_standard_edges}
            inference_edges = {
                score: {frozenset(edge) for edge in edges}
                for score, edges in inference_edges.items()
            }
            universe = {frozenset(edge) for edge in universe}
            # The size of the universe should not change, repetition are considered in __universe

        # Universe is used as reference
        edge_to_id = {edge: i for i, edge in enumerate(universe)}
        gold_standard_edges = {edge_to_id[edge] for edge in gold_standard_edges}
        # The get method is used to only keep edges between genes in the gold standard geneset
        # TODO: UX: The user may want to know the fraction of the inference used for the evaluation
        inference_edges = sorted(
            [
                [score, {edge_to_id.get(edge) for edge in edges}]
                for score, edges in inference_edges.items()
            ],
            reverse=self.greater_is_better,
        )

        # erase universe, mapping and gold standard geneset
        del universe
        del edge_to_id  # TODO: UX: user may want to keep this mapping
        del gold_standard_geneset
        # call garbage collector
        gc.collect()

        return gold_standard_edges, inference_edges, size_universe

    def __compute_roc_pr_datapoints(self, cutoff=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the false positive rate, sensitivity and precision for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None (default) the cutoff provided in the initialization is used. Use False to avoid using a cutoff.
            The cutoff follows the greater_is_better rule.

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
        cutoff = self.cutoff if cutoff is None else cutoff
        cache = True if cutoff == self.cutoff else False

        if (
            cache
            and self.precision is not None
            and self.sensitivity is not None
            and self.fpr is not None
        ):
            # print("The evaluation metrics have already been computed for this cutoff. Returning cached values.")
            return self.precision, self.sensitivity, self.fpr

        if cutoff is False:
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
        fpr = np.empty(num_points)
        precision = np.empty(num_points)
        sensitivity = np.empty(num_points)  # same as recall and TPR

        # Start-values
        fpr[0] = 0
        precision[0] = 0
        sensitivity[0] = 0

        predicted_positives = set()
        for i, edges in enumerate(inference_edges, 1):
            predicted_positives.update(edges)
            true_positives = len(self.gold_standard_edges & predicted_positives)
            false_positives = len(predicted_positives - self.gold_standard_edges)
            # remplace the corresponding values in the arrays
            fpr[i] = false_positives / self.size_negatives
            sensitivity[i] = true_positives / self.size_gold_standard
            precision[i] = true_positives / (true_positives + false_positives)

        # End-values
        # At the last step, every edge is considered as a positive by inference
        self.__precision_baseline = (
            self.size_gold_standard / self.size_universe
        )  # (GS/(GS + (Universe-GS)))
        precision[-1] = self.precision_baseline
        sensitivity[-1] = 1  # no FN (no negatives)
        fpr[-1] = 1  # no TN (no negatives)

        # The first precision values must equal the second one
        precision[0] = precision[1]

        if cache:
            # enters only when the three arrays are None
            self.__precision = precision
            self.__sensitivity = sensitivity
            self.__fpr = fpr

        return precision, sensitivity, fpr

    def __compute_auc(self, x, y) -> float:
        """Computes the area under the curve using the trapezoidal rule."""
        return np.trapz(x=x, y=y, dx=0.01)

    def area_under_precision_recall_curve(self, cutoff=None) -> float:
        """Computes the area under the precision-recall curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used. Use False to avoid using a cutoff.
            The cutoff follows the greater_is_better rule.

        Returns:
        -------
        area: float
            Area under the precision-recall curve.
        """
        precision, sensitivity, _ = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        # print(precision, sensitivity)
        return self.__compute_auc(x=sensitivity, y=precision)

    def area_under_roc_curve(self, cutoff=None) -> float:
        """Computes the area under the ROC curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the cutoff provided in the initialization is used. Use False to avoid using a cutoff.
            The cutoff follows the greater_is_better rule.

        Returns:
        -------
        area: float
            Area under the ROC curve.
        """
        _, sensitivity, fpr = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        return self.__compute_auc(x=fpr, y=sensitivity)

    def __plot_curve(self, x, y, xlabel, ylabel, title="AUC", ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        # Graph
        ax.set_facecolor(FACE_COLOR)
        ax.fill_between(x, y, color=MAIN_PLOT_COLOR, alpha=0.5)
        ax.plot(x, y, color=MAIN_PLOT_COLOR, alpha=1)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)

        # Add titles
        ax.set_title(title, loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_xlabel(xlabel, size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_ylabel(ylabel, size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax

    def plot_precision_recall_curve(self, cutoff=None, ax=None, **kwargs):
        """Plots the precision-recall curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the evaluation metrics are computed for every score in the inference.
            The cutoff follows the greater_is_better rule.
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
            title=f"AUC-PR = {self.__compute_auc(x=sensitivity, y=precision):.3f}",
            ax=ax,
        )

    def plot_roc_curve(self, cutoff=None, ax=None, **kwargs):
        """Plots the ROC curve.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            If None, the evaluation metrics are computed for every score in the inference.
            The cutoff follows the greater_is_better rule.
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
            self.fpr,
            self.sensitivity,
            xlabel="False positive rate",
            ylabel="True positive rate",
            title=f"AUC-ROC = {self.__compute_auc(x=fpr, y=sensitivity):.3f}",
            ax=ax,
        )

    # TODO: Optimization: Create a cache to keep {(cutoff, metric): value)}
    def recall(self, cutoff) -> float:
        """Computes the recall for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            The cutoff follows the greater_is_better rule.

        Returns:
        -------
        recall: float
            Recall for the given cutoff.
        """
        _, sensitivity, _ = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        return sensitivity[-2]  # last value is always 1

    def overall_precision(self, cutoff) -> float:
        """Computes the precision for a given cutoff.

        Args:
        cutoff (float): Cutoff to use to compute the evaluation metrics.
            The cutoff follows the greater_is_better rule.

        Returns:
        -------
        precision: float
            Precision for the given cutoff.
        """
        precision, _, _ = self.__compute_roc_pr_datapoints(cutoff=cutoff)
        return precision[-2]  # last value is always the baseline

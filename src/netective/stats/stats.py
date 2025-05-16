from __future__ import annotations
import os
import math
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations, product
from typing import Tuple
from bitarray import bitarray
import numpy as np
from warnings import warn
from networkx import Graph, DiGraph
from networkx.classes.function import selfloop_edges
from networkx.readwrite.edgelist import read_edgelist

from netective.logging_info import get_logger, set_log_level
stats_logger = get_logger(__name__)

# TODO: SENT TO GLOBALS.PY
FONT_SIZE = 11
FACE_COLOR = "#F2F0F2"
FONT_COLOR = "#060307"
MAIN_PLOT_COLOR = "#031926"
MINOR_FONT_COLOR = "#504B51"

class NullGraphError(Exception):
    """Exception raised for null graph."""
    pass

def remove_self_loops(G: DiGraph):
    G.remove_edges_from(selfloop_edges(G))
    return G

def parse_network(file_path: str, comments:str= "#", delimiter:str="\t", directed:bool= True, score:bool= False, use_position_as_score:bool= False, greater_score_is_better: bool=True) -> Union[DiGraph, Graph]:
    """Useful fxn for parsing a network file

    Fxn for parsing a network file, robust for several common file formats. It is also robust to ranking of edges when providing an edgelist with scores.

    Arguments:
        file_path (str): path to the network file.
        comments (str): comment character. Defaults to "#".
        delimiter (str): delimiter character. Defaults to "\t".
        directed (bool): whether the network will be created with nx.Graph or nx.DiGraph. Defaults to True.
        score (bool): if True, the network will use the third column of the file as the score of the edge. Defaults to False.
        use_position_as_score (bool): if True, the position of the edge in the file will be used as the score of the edge.. Defaults to False.
        greater_score_is_better (bool): Whether the inference score is better when it is higher or lower.
        
    Raises:
        ValueError: _description_
        ValueError: _description_
        NullGraphError: _description_

    Returns:
        Union[nx.DiGraph, nx.Graph]: networkx object."""

    if score and use_position_as_score:
        stats_logger.critical("score and use_position_as_score cannot be True at the same time.")
        raise ValueError("score and use_position_as_score cannot be True at the same time.")

    if not use_position_as_score:
        if score:
            # check if first line has 3 columns
            with open(file_path, "r") as f:
                first_line = f.readline()
                cols = first_line.strip().split(delimiter)
                if len(cols) < 3:
                    stats_logger.critical(
                        f"File {file_path} does not have a score column. Set score=False."
                    )
                    raise ValueError(
                        f"File {file_path} does not have a score column. Set score=False."
                    )

        G = read_edgelist(
            file_path,
            comments=comments,
            delimiter=delimiter,
            create_using=DiGraph if directed else Graph,
            data=(("score", float),) if score else False,
        )
    else:
        G = DiGraph() if directed else Graph()
        with open(file_path, "r") as f:
                edges = [line for line in f if not line.startswith(comments)]

                for i, line in enumerate(edges if not greater_score_is_better else edges[::-1]):
                    cols = line.strip().split(delimiter)
                    source = cols[0]
                    target = cols[1]

                    score = i

                    G.add_edge(source, target, score=score)
  
    if G.number_of_edges() == 0:
        stats_logger.critical(f'Empty graph detected after parsing. It is probably due to an error declaring delimiters or comments in network file -> {file_path}')
        raise NullGraphError(f'Empty graph detected after parsing. It is probably due to an error declaring delimiters or comments in network file -> {file_path}')
    
    return G

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
    """Returns the universe of potential edges between genes in the Gold Standard

    The universe is compute following the rules:
        - If the inference is directed and self-loops are allowed, the universe is the cartesian product of the gold standard geneset.
        - If the inference is directed and self-loops are not allowed, the universe is the permutations of the gold standard geneset.
        - If the inference is undirected and self-loops are allowed, the universe is the combinations with replacement of the gold standard geneset.
        - If the inference is undirected and self-loops are not allowed, the universe is the combinations without replacement of the gold standard geneset.
    See Combinatory iterators from itertools https://docs.python.org/3/library/itertools.html for info

    Arguments:
        gold_standard_geneset (set): set of unique genes present in the Gold Standard
        directed (bool): whether or not benchmarking will consider direction of interactions
        allow_self_loops (bool): whether or not benchmarking will allow self-regulations

    Returns:
        set[tuple[str, str]]: potential edges computed for benchmarking"""
    stats_logger.debug('Establishing universe from Gold Standard...')
    if directed and allow_self_loops:
        return set(product(gold_standard_geneset, repeat=2))
    elif directed and not allow_self_loops:
        return set(permutations(gold_standard_geneset, 2))
    elif not directed and allow_self_loops:
        return set(combinations_with_replacement(gold_standard_geneset, 2))
    else:  # if not directed and not allow_self_loops:
        return set(combinations(gold_standard_geneset, 2))

def _anonymize_inference_edges(inference, gold_standard_geneset, greater_score_is_better, edge_to_id, directed):
        stats_logger.info('Anonymizing inference edges')
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
    stats_logger.info('Anonymizing GS edges')
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
    gc.collect()

    return gold_standard_edges, inference_edges, size_universe

def benchmarking(
    networks: dict | str,
    gold_standard: Graph | DiGraph | str,
    directed: bool = True,
    greater_score_is_better: bool = True,
    allow_self_loops: bool = False,
    cutoff: float | False = False,
    optimal_cutoff: bool = False,
    f1_score: bool = False,
    mcc: bool = False,
    return_auc_coords_dicts: bool = False,
    comments : str = '#',
    delimiter : str = '\t',
    score : bool = True,
    verbose: str = None,
    baseline: bool = True,
) -> tuple | dict:
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
        return_auc_coords_dicts (bool): Whether to return the AUC values for every inference in the benchmark and their precision, sensitivity and fpr distributions respectively.
            If False, the figure axis are returned.
        comments (str): Character used to indicate comments in the network files.
        delimiter (str): Character used to separate the columns in the network files.
        score (bool): Whether the inference networks have a score attribute or not.
            If score is False, the ranking of the edges is used as score (e.g., the first edge has a score of 0, the second a score of 1, etc.).
        verbose (str): Level of verbosity.
            If None, the verbosity level is set to WARNING. Options are: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        baseline (bool): Whether to include a baseline inference in the benchmark.

    Returns:
        If return_auc_coords_dicts is True, a tuple containing the AUPRs, AUROCs, and (precision, recall, and FPR) for every inference in the benchmark.
        If return_auc_coords_dicts is False, a tuple containing the figure axis for the AUPR and AUROC plots.
    """

    if verbose is not None:
        current_level = stats_logger.getEffectiveLevel()
        set_log_level(stats_logger, verbose)

    stats_logger.info('Beginning benchmarking of inference(s)')

    if isinstance(networks, str):

        # valid networks dir and gold standard path
        stats_logger.debug('Parsing networks from directory...')
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
        stats_logger.debug('Validating inputed networks objects...')
        if gold_standard in networks:
            gold_standard = networks.pop(gold_standard)
        else:
            if not isinstance(gold_standard, (Graph, DiGraph)):
                raise TypeError("gold_standard must be a networkx.Graph or networkx.DiGraph.")
            gold_standard = gold_standard.copy()

    else:
        raise TypeError("networks must be a dictionary or a path to a directory.")
    
    if baseline:
        # include empty inference
        networks = {**networks, "Baseline": DiGraph() if gold_standard.is_directed() else Graph()}
    
    benchmark = Benchmark(
        gold_standard=gold_standard,
        inferences=networks,
        greater_score_is_better=greater_score_is_better,
        allow_self_loops=allow_self_loops,
        cutoff=cutoff,
        baseline=baseline,
    )

    if return_auc_coords_dicts:
        stats_logger.info('Calculating AUPR, AUROC, coordinates and F1 score distributions')
        return_set = benchmark.aupr, benchmark.auroc, benchmark.coordinates, benchmark.f1_score_dists, benchmark.mcc_dists, benchmark.accuracy_dists, benchmark.summarize
    
    else:
        return_set = {}
        stats_logger.info('Plotting AUPR and AUROC curves...')
        return_set['aupr'] = benchmark.plot_aupr()
        return_set['pr curves'] = benchmark.plot_precision_recall_curves()
        return_set['auroc'] = benchmark.plot_auroc()
        return_set['roc curves'] = benchmark.plot_roc_curves()
        if optimal_cutoff:
            return_set['optimal cutoffs'] = benchmark.plot_optimal_cutoffs()
        if f1_score:
            return_set['f1 scores'] = benchmark.plot_f1_score()
        if mcc:
            return_set['mcc'] = benchmark.plot_mcc()

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
        baseline: bool = True,
    ):
        """_summary_

        _extended summary_[#_unique ID_]_

        .. math:: _LaTeX formula_

        Arguments:
            gold_standard (Graph | DiGraph): _description_
            inferences (dict[str, Graph  |  DiGraph]): _description_
            greater_score_is_better (bool): _description_. Defaults to True.
            allow_self_loops (bool): _description_. Defaults to False.
            cutoff (float | False): _description_. Defaults to False.
            include_optimal_cutoff (bool): _description_. Defaults to False.
            baseline (bool): _description_. Defaults to True.

        References:
            .. [#_unique ID_] *_pubmed abbr journal title_* _vol_:_page or e-article id_ (_year_) https://doi.org/_doi_
            .. [#_unique ID_] _first-author first-name last-name_ *_book title_* (_year_) ISBN:_ISBN_ _http link_
            .. [#_unique ID_] _article title_ _conference_ (_year_) _http link_"""
        self.__repr_str = f"Benchmark({gold_standard}, {inferences}, greater_is_better={greater_score_is_better}, allow_self_loops={allow_self_loops}, cutoff={cutoff})"
        
        if baseline: # include empty inference
            inferences = {**inferences, "Baseline": DiGraph() if gold_standard.is_directed() else Graph()}

        if not allow_self_loops:
            stats_logger.debug('Removing self-loops from GS and inferences...')
            gold_standard = remove_self_loops(gold_standard)
            inferences = {net_id: remove_self_loops(inf) for net_id, inf in inferences.items()}

        gold_standard_edges, inference_edges, size_universe = _anonymize_edges(
            gold_standard,
            list(inferences.values()),
            greater_score_is_better,
            allow_self_loops,
        )

        stats_logger.debug('Creating LinkEval instances for every GS-inference pair...')
        self.__nis_instances = {
            net_id: LinkEval(
                cutoff=cutoff,
                gold_standard_edges=gold_standard_edges,
                inference_edges=inf,
                size_universe=size_universe,
                greater_score_is_better= greater_score_is_better,
                inference_id= net_id
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
        stats_logger.info('Plotting AUROC curves for every inference in the benchmark')
        ax = _build_ax(ax, figsize=(3,3))
        best_name, _ = self.best_auroc
        for net_id, nis in self.nis_instances.items():
            nis.plot_roc_curve(ax=ax, label=net_id if net_id==best_name else None, color='r' if net_id==best_name else MAIN_PLOT_COLOR, alpha=0, title=False, **kwargs)
        ax.legend()
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
        stats_logger.info('Plotting AUPR curves for every inference in the benchmark')
        ax = _build_ax(ax, figsize=(3,3))
        best_name, _ = self.best_aupr
        ylimit = max([nis.precision_dist[0] for _, nis in self.nis_instances.items()])
        for net_id, nis in self.nis_instances.items():
            nis.plot_precision_recall_curve(ax=ax, ylimit= ylimit, label=net_id if net_id==best_name else None, color='r' if net_id==best_name else MAIN_PLOT_COLOR, alpha=0, title=False, **kwargs)
        ax.legend()
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
        stats_logger.info('Plotting AUPR values for every inference in the benchmark')
        ax = _build_ax(ax, xlim=None, ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.area_under_precision_recall_curve() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("AUPR", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("AUPR", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        # ax.set_xlim(0, max([nis.area_under_precision_recall_curve() for nis in self.nis_instances.values()]))
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_auroc(self, ax=None, **kwargs):
        """Plots the AUROC values for every inference in the benchmark."""
        stats_logger.info('Plotting AUROC values for every inference in the benchmark')
        ax = _build_ax(ax, xlim=None, ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.area_under_roc_curve() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("AUROC", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("AUROC", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        # ax.set_xlim(0, max([nis.area_under_roc_curve() for nis in self.nis_instances.values()]))
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_f1_score(self, ax=None, **kwargs):
        """Plots the F1 score values for every inference in the benchmark."""
        stats_logger.info('Plotting F1 scores values for every inference in the benchmark')
        ax = _build_ax(ax, xlim=None, ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.f1_score() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("F1 score", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("F1 score", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        # ax.set_xlim(0, max([nis.f1_score() for nis in self.nis_instances.values()]))
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    def plot_mcc(self, ax=None, **kwargs):
        """Plots the MCC values for every inference in the benchmark."""
        stats_logger.info('Plotting MCC values for every inference in the benchmark')
        ax = _build_ax(ax, xlim=[-1.01,1.01], ylim=None)
        ax.barh(list(self.nis_instances.keys()), [nis.mcc() for nis in self.nis_instances.values()], color=MAIN_PLOT_COLOR, **kwargs)
        ax.set_title("MCC", loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax.set_ylabel("Inference", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.set_xlabel("MCC", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        return ax
    
    @property
    def best_aupr(self) -> tuple[str, float]:
        """Returns the inference with the best AUPR and its value."""
        return max([(net_id, nis.area_under_precision_recall_curve()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def best_auroc(self) -> tuple[str, float]:
        """Returns the inference with the best AUROC and its value."""
        return max([(net_id, nis.area_under_roc_curve()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])

    @property
    def best_f1_score(self) -> tuple[str, float]:
        """Returns the inference with the best F1 score and its value."""
        return max([(net_id, nis.f1_score()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def best_accuracy(self) -> tuple[str, float]:
        """Returns the inference with the best accuracy and its value."""
        return max([(net_id, nis.accuracy()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def best_recall(self) -> tuple[str, float]:
        """Returns the inference with the best recall and its value."""
        return max([(net_id, nis.recall()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def best_precision(self) -> tuple[str, float]:
        """Returns the inference with the best precision and its value."""
        return max([(net_id, nis.precision()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def best_mcc(self) -> tuple[str, float]:
        """Returns the inference with the best MCC and its value."""
        return max([(net_id, nis.mcc()) for net_id, nis in self.nis_instances.items()], key=lambda x: x[1])
    
    @property
    def aupr(self) -> dict[str, float]:
        """Returns the AUPR values for every inference in the benchmark."""
        return {net_id: nis.area_under_precision_recall_curve() for net_id, nis in self.nis_instances.items()}
    
    @property
    def auroc(self) -> dict[str, float]:
        """Returns the AUROC values for every inference in the benchmark."""
        return {net_id: nis.area_under_roc_curve() for net_id, nis in self.nis_instances.items()}
    
    @property
    def coordinates(self, cutoff=None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns the coordinates for the precision, recall and FPR for every inference in the benchmark."""
        return {net_id: nis.coordinates(cutoff) for net_id, nis in self.nis_instances.items()}
    
    @property
    def f1_score_dists(self)-> dict[str, dict[float, float]]:
        """Returns the F1 distribution for every inference in the benchmark."""
        return {net_id: nis.compute_f1_score_dist() for net_id, nis in self.nis_instances.items()}
    
    @property
    def mcc_dists(self) -> dict[str, dict[float, float]]:
        """Returns the MCC distribution for every inference in the benchmark."""
        return {net_id: nis.score_mcc_dist for net_id, nis in self.nis_instances.items()}
    
    @property
    def accuracy_dists(self) -> dict[str, dict[float, float]]:
        """Returns the accuracy distribution for every inference in the benchmark."""
        return {net_id: nis.score_accuracy_dist for net_id, nis in self.nis_instances.items()}
    
    @property
    def summarize(self) -> pd.DataFrame:
        """Summarizes the evaluation metrics for every inference in the benchmark.

        Returns:
        -------
        summary: pd.DataFrame
            DataFrame containing the evaluation metrics for every inference in the benchmark.
        """
        stats_logger.info('Creating stats summary...')
        summary = {
            net_id: {
                "AUPR": nis.area_under_precision_recall_curve(),
                "AUROC": nis.area_under_roc_curve(),
                "F1 score": nis.f1_score() if nis.inference_id != 'Baseline' else np.nan,
                "MCC": nis.mcc() if nis.inference_id != 'Baseline' else np.nan,
                "Optimal cutoff": nis.optimal_cutoff() if nis.inference_id != 'Baseline' else np.nan,
                "Accuracy": nis.accuracy() if nis.inference_id != 'Baseline' else np.nan
            }
            for net_id, nis in self.nis_instances.items()
        }
        return pd.DataFrame(summary).T

class LinkEval:
    def __init__(
        self,
        gold_standard: Graph | DiGraph = None,
        inference: Graph | DiGraph = None,
        greater_score_is_better: bool = True,
        allow_self_loops: bool = False,
        cutoff: float | False = False,
        gold_standard_edges: set = None,
        inference_edges: dict = None,
        size_universe: int = None,
        inference_id: str = None
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
        self.__inference_id = inference_id

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

        stats_logger.info('Establishing Gold Standard size, Negatives size and Precision baseline')
        self.__size_gold_standard = len(self.__gold_standard_edges)
        self.__size_negatives = self.__size_universe - self.__size_gold_standard
        # At the last step, every edge is considered as a positive by inference
        self.__precision_baseline = (self.size_gold_standard / self.size_universe)  # (GS/(GS + (Universe-GS)))
        if not cutoff:
            try:
                stats_logger.warning('No cutoff provided, establishing one based on greater_is_better argument...')
                self.__cutoff = min([score for score, _ in self.inference_edges]) if self.greater_is_better else max([score for score, _ in self.inference_edges])
            except ValueError:
                if inference_id == 'Baseline': # Empty inference is only considered for Baselines
                    stats_logger.warning(f"The inference: {inference_id} is empty. No F1 scores, optimal cutoff or Matthews Correlation Coefficient will be computed.")
                    self.__cutoff = None
                else: # Gold Standard input is probably wrong for input inference, or viceversa
                    stats_logger.critical(f'The inference {inference_id} is empty. This could be caused because Gold Standard does not match inference in origin. Netective eliminates edges from the inference if nodes not present in the Gold Standard are detected.')
                    raise ValueError(f'The inference {inference_id} is empty. This could be caused because Gold Standard does not match inference in origin. Netective eliminates edges from the inference if nodes not present in the Gold Standard are detected.')
        else:
            self.__cutoff = cutoff

        # Define evaluation
        # Used as flags to know if the curves data points have been computed
        self.__fpr_dist = None
        self.__score_fpr_dist = None
        self.__precision_dist = None
        self.__score_pre_dist = None
        self.__sensitivity_dist = None
        self.__score_sensi_dist = None
        self.__f1_score_dist = None
        self.__mcc_dist = None
        self.__accuracy_dist = None

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

    # TODO is it really necessary to delete cache?
    @cutoff.setter
    def cutoff(self, cutoff: float | None):
        stats_logger.warning('New cutoff established. Cache of previous datapoints cleared.')
        self.__cutoff = cutoff
        # Reset the cache
        self.__precision_dist = None
        self.__score_pre_dist = None
        self.__sensitivity_dist = None
        self.__score_sensi_dist = None
        self.__fpr_dist = None
        self.__score_fpr_dist = None 
        self.__f1_score_dist = None
        self.__mcc_dist = None
        self.__accuracy_dist = None

    @property
    def precision_dist(self) -> np.ndarray:
        """Returns precision datappoints

        Returns:
            np.ndarray: precision datapoints for pr curve"""
        return self.__precision_dist
    
    @property
    def score_precision_dist(self) -> dict:
        """Returns precision distribution for every score in the inference

        Returns:
            dict: dictionary with scores as keys and precision for that score as less restrictive cutoff as values"""
        return self.__score_pre_dist
    
    @property
    def sensitivity_dist(self) -> np.ndarray:
        """Returns sensitivity (recall) datapoints

        Returns:
            np.ndarray: sensitivity datapoints for pr and roc curves"""
        return self.__sensitivity_dist
    
    @property
    def score_sensitivity_dist(self) -> dict:
        """Returns sensitivity (recall) distribution for every score in the inference
        
        Returns:
            dict: dictionary with scores as keys and sensitivity for that score as less restrictive cutoff as values"""
        return self.__score_sensi_dist

    @property
    def fpr_dist(self) -> np.ndarray:
        """Returns false positive rate datapoints

        Returns:
            np.ndarray: false positive rate for roc curve"""
        return self.__fpr_dist
    
    @property
    def score_fpr_dist(self) -> dict:
        """Returns false positive rate for every score in the inference

        Returns:
            dict: dictionary with scores as keys and false positive rate for that score as less restrictive cutoff as values"""
        return self.__score_fpr_dist
    
    @property
    def f1_scores_dist(self) -> dict:
        """Returns F1 scores distribution

        Returns:
            dict: dictionary with scores as keys and the F1 score calculated from that score's Precision and Recall"""
        return self.__f1_score_dist
    
    @property
    def score_mcc_dist(self) -> dict:
        """Returns Matthews Correlation Coefficient distribution

        Returns:
            dict: dictionary with scores as keys and MCC for that score as less restrictive cutoff as values"""
        return self.__mcc_dist

    @property
    def score_accuracy_dist(self) -> dict:
        """Returns the Accuracy distribution

        Returns:
            dict: dictionary with scores as keys and the accuracy calculated for that score as less restrictive cutoff as values"""
        return self.__accuracy_dist

    @property
    def size_universe(self) -> int:
        """Returns the size of the universe

        The universe is composed by, according to two parameters passed as args:
            directed & allow_self_loops: product
            directed & !allow_self_loops: permutations
            !directed & allow_self_loops: combinations with replacements
            !directed & !allow_self_loops: combinations
        See Combinatory iterators at https://docs.python.org/3/library/itertools.html for info on each iterator

        Returns:
            int: size of Universe used as reference in benchmarking"""
        return self.__size_universe
    
    @property
    def precision_baseline(self) -> float:
        """Returns the precision baseline

        .. math:: Precision Baseline = \frac{\left\| Gold Standard \right\|}{\left\| Universe \right\|}

        Returns:
            float: precision baseline as a float"""
        return self.__precision_baseline

    @property
    def size_negatives(self) -> int:
        """Returns the size of the Negatives

        .. math:: \left\| Negatives \right\| = \left\| Universe \right\| - \left\| Gold Standard \right\|

        Returns:
            int: number of negatives in reference to Universe and Gold Standard sizes"""
        return self.__size_negatives

    @property
    def size_gold_standard(self) -> int:
        """Returns size of Gold Standard
    
        Returns:
            int: number of edges in Gold Standard"""
        return self.__size_gold_standard

    @property
    def gold_standard_edges(self) -> set:
        """Returns Gold Standard edges

        Anonymized (from the Universe) edges from Gold Standard

        Returns:
            set: anonymized set of edges"""
        return self.__gold_standard_edges

    @property
    def inference_edges(self) -> dict[float, set]:
        """Returns inference edges

        Anonymized (from the Universe) edges from the inference.
        They are sorted according to the arg greater_is_better from most restrictive to less restrictive.
        All edges with the same score are grouped together in the same set.

        Returns:
            dict[float, set]: dictionary with the inference's unique scores as keys and the set of edges with said score as values"""
        return self.__inference_edges

    @property
    def gold_standard(self) -> Graph | DiGraph:
        """Returns Gold Standard network

        Returns:
            Graph | DiGraph: either a networkx Graph or DiGraph object"""
        return self.__gold_standard

    @property
    def inference(self) -> Graph | DiGraph:
        """Returns inference network

        Returns:
            Graph | DiGraph: either a networkx Graph or DiGraph object"""
        return self.__inference

    @property
    def greater_is_better(self) -> bool:
        """Returns whether less restrictive behavior of cutoff

        If greater is better, the higher the score, the better.
        Viceversa, if False, the lower the score the better.

        Returns:
            bool: flag that controls less restrictive behavior of cutoff"""
        return self.__greater_is_better

    @property
    def directed(self) -> bool:
        """Returns whether Gold Standard and inference consider direction

        Returns:
            bool: flag that indicates consideration of direction in benchmarking"""
        return self.__directed

    @property
    def allow_self_loops(self) -> bool:
        """Returns whether the Gold Standard and inference allow self-loops

        Returns:
            bool: flag that indicates consideration of self-loops in benchmarking"""
        return self.__allow_self_loops
    
    @property
    def inference_id(self) -> str:
        """Returns the inference name

        Returns:
            str: inference ID as a string"""
        return self.__inference_id
    
    def __validate_networks(self):
        for name, network in [("Gold standard", self.gold_standard), ("Inference", self.inference)]:
            if not isinstance(network, (Graph, DiGraph)):
                stats_logger.critical(f'Network {name} provided is not a compatible Networkx object.')
                raise TypeError(f"{name} must be a networkx.Graph or networkx.DiGraph.")

        if self.gold_standard.is_directed() != self.inference.is_directed():
            stats_logger.critical('Gold standard and inference must be both directed or undirected.')
            raise ValueError("Gold standard and inference must be both directed or undirected.")

        if not all("score" in data for _, _, data in self.inference.edges(data=True)):
            stats_logger.critical("Inference edges must have a score attribute.")
            raise ValueError("Inference edges must have a score attribute.")

        if any("score" in data for _, _, data in self.gold_standard.edges(data=True)):
            warn("The gold standard edges have a score attribute. It might be a prediction.")

    def __get_tp_fp(self, gs_edges: set, inference_edges: dict) -> list[tuple[int, int]]:
        """Calculates true positives and true negatives in reference to Gold Standard

        Optimized implementation that performs set operations using bitwise operators.
        The idea behind this method, is converting the Universe (union of Gold Standard edges and inference edges) to
        an array of bits (using bitarray objects). From this "coded universe", we create a bitwise array of the same length
        for both the Gold Standard edges and inference edges with all 0s. Here each cell in the array represents one element in the "coded
        universe", so if the element exists in the set of Gold Standard edges, a 1 in storaged in that position. As a result,
        we have a "coded array" for the Gold Standard edges that can be used for set operations and could later be mapped back
        to the original IDs of the elements. The same is done for the inference edges, with the distinction that this array
        grows N elements each iteration. The number of iterations is the number of unique scores the inference has and each N
        represents all edges associated to that unique score.
        With both "coded" arrays, the intersection is simply a bitwise & (AND) operator. All set operations can be performed
        using bitwise operators, but for this implementation only the intersection and assymetric differences (performed as an intersection)
        are necessary for the computation of true positives and false positives.

        For more info on the bitarray objects used, read the library documentation: https://pypi.org/project/bitarray/

        .. math:: tp = N(Gold Standard \cap Inference)
        .. math:: fp = N(Inference \cap \sim tp)

        Arguments:
            gs_edges (set): anonimyzed edges from the Gold Standard.
            inference_edges (dict): dictionary with the anonimyzed edges from the inference. Keys are each unique score in the
                                    inference and all edges associated to that score as values.

        Returns:
            list[tuple[int, int]]: list with n number of tuples, where n is the number of unique scores in the inference
                                   and each tupple is that score's true positives and false positives"""
        stats_logger.debug(f'Calculating true positives and false positives for: {self.__inference_id}')
        # Getting complete set of predicted edges
        all_predicted_edges = set()
        for edges in inference_edges:
            all_predicted_edges.update(edges)
        
        # Universe creation
        stats_logger.debug('Creating universe for binary mapping...')
        universe = gs_edges | all_predicted_edges
        ids_indexes = {
            ids : i
            for i, ids in enumerate(universe)
        }

        # Binary Gold Standard Edges array
        stats_logger.debug('Creating binary array of GS edges mapped to a binary universe of both GS and inference edges.')
        gs_bin = bitarray(len(universe))
        for interaction in gs_edges:
            gs_bin[ids_indexes[interaction]] |= 1

        # Predicted Positives Edges array & true positives and false positives calculations
        tp_fp = []
        stats_logger.debug(f'{self.__inference_id} has a total of {len(inference_edges)} scores after trimming.')
        predicted_positives_bin = bitarray(len(universe))
        for i, bunch_edges in enumerate(inference_edges):
            stats_logger.debug(f"Creating {i + 1}/{len(inference_edges)} binary array and calculating this score's true positives and false positives...")
            
            # Predicted edges binary array update
            for interaction in bunch_edges:
                predicted_positives_bin[ids_indexes[interaction]] |= 1
            
            # Intersection
            intersection_bin = gs_bin & predicted_positives_bin

            # Intersection cardinality
            tp = intersection_bin.count(1)

            # Substraction cardinality
            fp = (predicted_positives_bin & ~intersection_bin).count(1)

            tp_fp.append((tp, fp))

        return tp_fp

    def __compute_roc_pr_datapoints(self, cutoff: float=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO add references to metrics calculations
        """Computes the false positive rate, sensitivity and precision for a given cutoff.

        It caches the values for the given cutoff if the cutoff is the same as the one provided in the initialization.
        The inference edges must be sorted by score following the greater_is_better rule.
        It returns the cache values if the cutoff is the same as the one provided in the initialization or if cutoff is None.
        It also calculates the Matthews Correlation Coefficient and accuracy for every score in the inference. Thess are
        storaged as internal attributes of the class, wich the user can access through a public class method. They are not returned
        by this method.
        This method returns the Precision, Sensitivity and FPR datapoints used for the PR and ROC curves, the actual distributions
        are storaged in an internal attribute of the class, accesible to the user through public class methods specific for each metric.

        .. math:: precision = \frac{tp}{tp + fp}
        .. math:: sensitivity = sensitivity = \frac{tp}{\left\| Gold Standard \right\|}
        .. math:: fpr = \fpr = \frac{fp}{\left\| Negatives \right\|}
        .. math:: mcc = mcc = \frac{tp*tn + fp*fn}{\sqrt{(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)}}
        .. math:: accuracy = accuracy = \frac{tp+tn}{\left\| Universe \right\|}

        Arguments:
            cutoff (float): Cutoff to use to compute the evaluation metrics. Defaults to None.
                            If None the cutoff provided in the initialization is used.
                            The cutoff follows the greater_is_better rule and cannot be less restrictive than the one provided
                            in the initialization or the cutoff value set.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: tuple with the precision, sensitivity and fpr (as arrays) for that given cutoff."""
        
        cutoff = self.__validate_cutoff(cutoff)
        cache = True if cutoff == self.cutoff else False
        # TODO:!!! optimization: use a subset of the computed datapoints.
        if (
            cache
            and self.precision_dist is not None
            and self.sensitivity_dist is not None
            and self.fpr_dist is not None
        ):
            return self.precision_dist, self.sensitivity_dist, self.fpr_dist
        
        stats_logger.warning(f'Computing pr and roc datapoints for: {self.__inference_id} and cutoff: {cutoff}')
        stats_logger.debug(f'Trimming predicted edges according to cutoff established ({cutoff}) and greater_is_better argument ({self.greater_is_better})')
        if self.greater_is_better:
            inference_edges = [
                edges for score, edges in self.inference_edges if score >= cutoff
            ]
            ordered_scores = [
                score for score, edges in self.inference_edges if score >= cutoff
            ]
        else:
            inference_edges = [
                edges for score, edges in self.inference_edges if score <= cutoff
            ]
            ordered_scores = [
                score for score, edges in self.inference_edges if score <= cutoff
            ]

        num_points = len(ordered_scores) + 2
        # Initialize arrays to store the evaluation metrics coordinates
        fpr_dist = np.empty(num_points)
        precision_dist = np.empty(num_points)
        sensitivity_dist = np.empty(num_points)
        mcc_dist = np.empty(len(ordered_scores))
        accuracy_dist = np.empty(len(ordered_scores))

        # Start-values
        fpr_dist[0] = 0
        precision_dist[0] = 1
        sensitivity_dist[0] = 0

        # True positives and False positives from optimized intersection
        tp_fp = self.__get_tp_fp(
            gs_edges= self.gold_standard_edges,
            inference_edges= inference_edges
        )

        # Statistics calculation
        stats_logger.debug('Calculating precision, sensitivity and false positive rate for each score in the inference...')
        for i, elements in enumerate(tp_fp, 1):
            true_positives, false_positives = elements
            false_negatives = self.__size_gold_standard - true_positives
            true_negatives = self.__size_negatives - false_positives

            # Replace the corresponding values in the arrays
            fpr_dist[i] = false_positives / self.size_negatives
            sensitivity_dist[i] = true_positives / self.size_gold_standard
            precision_dist[i] = true_positives / (true_positives + false_positives)
            mcc_dist[i-1] = (true_positives * true_negatives - false_positives * false_negatives) / np.sqrt(float((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives)))
            accuracy_dist[i-1] = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

        # End-values
        precision_dist[-1] = self.precision_baseline
        sensitivity_dist[-1] = 1  # no FN (no negatives)
        fpr_dist[-1] = 1  # no TN (no negatives)

        # The first precision values must equal the second one
        precision_dist[0] = precision_dist[1]

        if cache:
            # enters only when the three arrays are None
            stats_logger.debug('Setting cache for computed datapoints...')
            self.__precision_dist = precision_dist
            self.__sensitivity_dist = sensitivity_dist
            self.__fpr_dist = fpr_dist
            self.__score_pre_dist = { score : precision for score, precision in zip(ordered_scores, precision_dist[1:-1]) }
            self.__score_sensi_dist = { score : sensitivity for score, sensitivity in zip(ordered_scores, sensitivity_dist[1:-1]) }
            self.__score_fpr_dist = { score : fpr for score, fpr in zip(ordered_scores, fpr_dist[1:-1]) }
            self.__mcc_dist = { score : mcc for score, mcc in zip(ordered_scores, mcc_dist) }
            self.__accuracy_dist = { score : accuracy for score, accuracy in zip(ordered_scores, accuracy_dist) }

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

    def __plot_curve(self, x, y, xlabel, ylabel, ylimit=None, title="AUC", ax=None, color=MAIN_PLOT_COLOR, alpha = 0.8, **kwargs):
        ax = _build_ax(ax, xlim=(0, 1.02), ylim=(0, 1.02))
        ax.fill_between(x, y, color=color, alpha=alpha)
        ax.plot(x, y, color=color, alpha=1, **kwargs)
        # Add titles
        if ylabel == 'Precision':
            ax.set_ylim(0, ylimit)
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
                stats_logger.critical(f'The cutoff must be greater than the one provided in the initialization ({self.cutoff}).')
                raise ValueError(
                    f"The cutoff must be greater than the one provided in the initialization ({self.cutoff})."
                )
        else:
            if cutoff is not None and cutoff > self.cutoff:
                stats_logger.critical(f"The cutoff must be lower or equal than the one provided in the initialization ({self.cutoff}).")
                raise ValueError(
                    f"The cutoff must be lower or equal than the one provided in the initialization ({self.cutoff})."
                )
        return cutoff

    def plot_precision_recall_curve(self, ylimit=None,cutoff=None, ax=None, title=True, **kwargs):
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
        if ylimit is None:
            ylimit = precision[0]
        ax = self.__plot_curve(
            sensitivity,
            precision,
            xlabel="Recall",
            ylabel="Precision",
            ylimit= ylimit,
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
        stats_logger.debug(f'Trimming predicted edges according to cutoff established ({cutoff}) and greater_is_better argument ({self.greater_is_better})')
        cutoff = self.__validate_cutoff(cutoff)
        if self.greater_is_better:
            return set().union(*[edges for score, edges in self.inference_edges if score >= cutoff])
        else:
            return set().union(*[edges for score, edges in self.inference_edges if score <= cutoff])

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
        stats_logger.debug(f'Calculating recall for specific cutoff: {cutoff}')

        if self.score_sensitivity_dist is not None:
            if cutoff in self.score_sensitivity_dist.keys():
                return self.score_sensitivity_dist[cutoff]
            else:
                closest_value = min(self.score_sensitivity_dist.keys(), key= lambda v:abs(v - cutoff))
                stats_logger.warning(f'Score: {cutoff} not available, showing result for closest score available: {closest_value}')
                return self.score_sensitivity_dist[closest_value]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            return true_positives / self.size_gold_standard

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
        stats_logger.debug(f'Calculating precision for cutoff: {cutoff}')
        
        if self.score_precision_dist is not None:
            if cutoff in self.score_precision_dist.keys():
                return self.score_precision_dist[cutoff]
            else:
                closest_value = min(self.score_precision_dist.keys(), key= lambda v:abs(v - cutoff))
                stats_logger.warning(f'Score: {cutoff} not available, showing result for closest score available: {closest_value}')
                return self.score_precision_dist[closest_value]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            return true_positives / (true_positives + false_positives)
    
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
        stats_logger.debug(f'Calculating false positive rate for cutoff: {cutoff}')

        if self.score_fpr_dist is not None:
            if cutoff in self.score_fpr_dist.keys():
                return self.score_fpr_dist[cutoff]
            else:
                closest_value = min(self.score_score_dfpr_dist.keys(), key= lambda v:abs(v - cutoff))
                stats_logger.warning(f'Score: {cutoff} not available, showing result for closest score available: {closest_value}')
                return self.score_fpr_dist[closest_value]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            return false_positives / self.size_negatives
    
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
        if self.inference_id == 'Baseline':
            return np.nan
        cutoff = self.__validate_cutoff(cutoff)
        stats_logger.info(f'Calculating accuracy for cutoff: {cutoff}')
        if self.score_accuracy_dist is not None:
            if cutoff in self.score_accuracy_dist.keys():
                return self.score_accuracy_dist[cutoff]
            else:
                closest_value = min(self.score_accuracy_dist.keys(), key= lambda v:abs(v - cutoff))
                stats_logger.warning(f'Score: {cutoff} not available, showing result for closest score available: {closest_value}')
                return self.score_accuracy_dist[closest_value]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            false_negatives = self.size_gold_standard - true_positives
            true_negatives = self.size_negatives - false_positives
            return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    
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
        if self.inference_id == 'Baseline':
            return np.nan
        cutoff = self.__validate_cutoff(cutoff)
        stats_logger.info(f'Calculating Matthews Correlation Coefficient for cutoff: {cutoff}')
        if self.score_mcc_dist is not None:
            if cutoff in self.score_mcc_dist.keys():
                return self.score_mcc_dist[cutoff]
            else:
                closest_value = min(self.score_mcc_dist.keys(), key= lambda v:abs(v - cutoff))
                stats_logger.warning(f'Score: {cutoff} not available, showing result for closest score available: {closest_value}')
                return self.score_mcc_dist[closest_value]
        else:
            predicted_edges = self.__filtered_inference_edges(cutoff=cutoff)
            true_positives = len(self.gold_standard_edges & predicted_edges)
            false_positives = len(predicted_edges - self.gold_standard_edges)
            false_negatives = self.size_gold_standard - true_positives
            true_negatives = self.size_negatives - false_positives
            return (true_positives * true_negatives - false_positives * false_negatives) / np.sqrt(float((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives)))
        
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
        if self.inference_id == 'Baseline':
            return np.nan
        cutoff = self.__validate_cutoff(cutoff)
        stats_logger.debug(f'Calculating F1 score for cutoff: {cutoff}')
        precision = self.precision(cutoff=cutoff)
        recall = self.recall(cutoff=cutoff)
        if precision+recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)
    
    def compute_f1_score_dist(self) -> dict:
        """Computes the F1 score for every score in the inference.

        Returns:
        f1_score_dist: dict[float, float]
            Dictionary containing the F1 score for every score in the inference.
        """
        if self.inference_id == 'Baseline':
            return {}
        
        if self.f1_scores_dist is None:
            if self.score_precision_dist is None:
                stats_logger.warning(f'Calculating Precision and Recall values for every score in the inference: {self.inference_id}')
                _ = self.area_under_precision_recall_curve()
            else:
                stats_logger.warning(f"Retrieving cached Precision and Recall to calculate each score in the inference's F1 score: {self.inference_id}")
            f1_score_dist = {}
            for score, _ in self.inference_edges:
                if (self.greater_is_better and score >= self.cutoff) or (not self.greater_is_better and score <= self.cutoff):
                    f1_score_dist[score] = self.f1_score(cutoff=score)
            self.__f1_score_dist = f1_score_dist
            return f1_score_dist
        else:
            return self.f1_scores_dist
    
    def optimal_cutoff(self) -> float:
        """Computes the optimal cutoff for the inference, that is required to maximize the F1 score.

        Returns:
        optimal_cutoff: float
            Optimal cutoff for the inference.
        """
        if self.inference_id == 'Baseline':
            return np.nan
        stats_logger.info('Calculating optimal cutoff for the inference. Requires to maximize F1 score.')
        self.__f1_score_dist = self.__f1_score_dist if self.__f1_score_dist is not None else self.compute_f1_score_dist()
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
        self.__f1_score_dist = self.__f1_score_dist if self.__f1_score_dist is not None else self.compute_f1_score_dist()
        optimal_cutoff = self.optimal_cutoff()
        if optimal_cutoff == self.cutoff:
            stats_logger.warning(f'Optimal cutoff is the same as the cutoff established.')
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
    
    @staticmethod
    def curves_from_coordinates(precision: list | np.ndarray, recall: list | np.ndarray, fpr: list | np.ndarray, ax=None, title: str=None, **kwargs) -> tuple[float, float]:
        # doesn't make much sense to make this a method of the class since it doesn't use any of the class attributes
        # but you cannot recover the lost information from the coordinates such as the cutoffs, the inference edges, etc.
        """
        Plot ROC and Precision-Recall curves and calculate AUCs from coordinates, acknowledging that recall equals TPR.

        Args:
            precision (list or np.ndarray): Precision values.
            recall (list or np.ndarray): Recall values, serving as TPR for the ROC curve and as recall for the Precision-Recall curve.
            fpr (list or np.ndarray): False Positive Rate values for the ROC curve.
            ax (matplotlib.axes.Axes): Axes object to plot the curve.
                If None, a new figure and axes are created.
            title (str): Title to add to the plot.
            **kwargs: Keyword arguments to pass to matplotlib.pyplot.plot.
        
        Returns:
        -------
        auc_pr: float
            Area under the Precision-Recall curve.
        auc_roc: float
            Area under the ROC curve.
            
        """

        # TODO: optimization. Use ax = self.__plot_curve
        ax_roc = _build_ax(ax, xlim=(0, 1.02), ylim=(0, 1.02))
        ax_roc.fill_between(fpr, recall, color=MAIN_PLOT_COLOR, alpha=1)
        ax_roc.plot(fpr, recall, color='k', alpha=1, **kwargs)
        
        if title is not None:
            ax_roc.set_title(title, loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax_roc.set_xlabel("False positive rate", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax_roc.set_ylabel("True positive rate", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax_roc.tick_params(axis="both", colors=MINOR_FONT_COLOR)
        
        ax_pr = _build_ax(ax, xlim=(0, 1.02), ylim=(0, 1.02))
        ax_pr.fill_between(recall, precision, color=MAIN_PLOT_COLOR, alpha=1)
        ax_pr.plot(recall, precision, color='k', alpha=1, **kwargs)
        
        if title is not None:
            ax_pr.set_title(title, loc="right", size=FONT_SIZE, color=FONT_COLOR)
        ax_pr.set_xlabel("Recall", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax_pr.set_ylabel("Precision", size=FONT_SIZE, color=MINOR_FONT_COLOR)
        ax_pr.tick_params(axis="both", colors=MINOR_FONT_COLOR)

        auc_pr = np.trapz(x=recall, y=precision, dx=0.05)
        auc_roc = np.trapz(x=fpr, y=recall, dx=0.05)
        return auc_roc, auc_pr, ax_roc, ax_pr

class BenchmarkPlotter(Benchmark):
    def __init__(
            self,
            stats_summary_df: pd.DataFrame,
            coords: dict,
            f1_score_dists: dict,
            mcc_score_dists: dict,
            accuracy_score_dists: dict,
            greater_is_better: bool = True
    ):
        """_summary_

        _extended summary_[#_unique ID_]_

        .. math:: _LaTeX formula_

        Arguments:
            stats_summary_df (pd.DataFrame): _description_
            coords (dict): _description_
            f1_score_dists (dict): _description_
            mcc_score_dists (dict): _description_
            accuracy_score_dists (dict): _description_
            greater_is_better (dict): _description_

        References:
            .. [#_unique ID_] *_pubmed abbr journal title_* _vol_:_page or e-article id_ (_year_) https://doi.org/_doi_
            .. [#_unique ID_] _first-author first-name last-name_ *_book title_* (_year_) ISBN:_ISBN_ _http link_
            .. [#_unique ID_] _article title_ _conference_ (_year_) _http link_"""
        self._Benchmark__repr_str = f"Benchmark(number_of_inferences={len(stats_summary_df.index)}, inferences_ids={stats_summary_df.index})"

        stats_logger.debug('Creating LinkEvalPlotter instances for every GS-inference pair...')
        nets_names = list(stats_summary_df.index)
        self._Benchmark__nis_instances = {}
        for net_name in nets_names:
            aupr_score = stats_summary_df.loc[net_name]['AUPR']
            auroc_score = stats_summary_df.loc[net_name]['AUROC']
            f1_score = stats_summary_df.loc[net_name]['F1 score']
            mcc = stats_summary_df.loc[net_name]['MCC']
            optimal_cutoff = stats_summary_df.loc[net_name]['Optimal cutoff']
            accuracy = stats_summary_df.loc[net_name]['Accuracy']
            # Score-metric distributions
            f1_score_dist = f1_score_dists[net_name] if net_name != 'Baseline' else None
            mcc_score_dist = mcc_score_dists[net_name] if net_name != 'Baseline' else None
            accuracy_score_dist = accuracy_score_dists[net_name] if net_name != 'Baseline' else None
            # Curves datapoints
            precision_dist = coords['precision'][net_name]
            sensitivity_dist = coords['sensitivity'][net_name]
            fpr_dist = coords['fpr'][net_name]

            self._Benchmark__nis_instances[net_name] = LinkEvalPlotter(
                aupr= aupr_score,
                auroc= auroc_score,
                precision_dist= precision_dist,
                sensitivity_dist= sensitivity_dist,
                fpr_dist= fpr_dist,
                f1_score= f1_score,
                mcc= mcc,
                accuracy= accuracy,
                f1_score_dist= f1_score_dist,
                mcc_score_dist= mcc_score_dist,
                accuracy_score_dist= accuracy_score_dist,
                optimal_cutoff= optimal_cutoff,
                greater_is_better= greater_is_better,
                inference_id= net_name
            )
    
    @property
    def best_recall(self) -> tuple[str, float]:
        """Returns the inference with the best recall and its value."""
        return max([(net_id, nis.recall()) for net_id, nis in self.nis_instances.items() if net_id != 'Baseline'], key=lambda x: x[1])
    
    @property
    def best_precision(self) -> tuple[str, float]:
        """Returns the inference with the best precision and its value."""
        return max([(net_id, nis.precision()) for net_id, nis in self.nis_instances.items() if net_id != 'Baseline'], key=lambda x: x[1])

class LinkEvalPlotter(LinkEval):
    def __init__(
        self,
        aupr: float,
        auroc: float,
        precision_dist: np.array,
        sensitivity_dist: np.array,
        fpr_dist: np.array,
        f1_score: float,
        mcc: float,
        accuracy: float,
        f1_score_dist: dict = None,
        mcc_score_dist: dict = None,
        accuracy_score_dist: dict = None,
        optimal_cutoff: float = None,
        greater_is_better: bool = True,
        inference_id: str = None
    ):
        """
        Class for visualizing previously computed benchmarking.

        Args:


        Notes:
            This class is unable to compute any statistical evaluation. It only visualizes previously computed
            benchmarking taking advantage of the LinkEval class's plotting method's.
        """
        # Class atributes that will not be used
        self._LinkEval__gold_standard = None
        self._LinkEval__inference = None
        self._LinkEval__allow_self_loops = None
        self._LinkEval__gold_standard_edges = None
        self._LinkEval__inference_edges = None
        self._LinkEval__size_universe = None
        self._LinkEval__directed = None
        self._LinkEval__size_gold_standard = None
        self._LinkEval__size_negatives = None
        self._LinkEval__precision_baseline = None

        # Class attributes that have been previously computed and provided by user
        self._LinkEval__inference_id = inference_id
        self._LinkEval__greater_is_better = greater_is_better
        # Curves datapoints
        self._LinkEval__fpr_dist = fpr_dist
        self._LinkEval__precision_dist = precision_dist
        self._LinkEval__sensitivity_dist = sensitivity_dist
        # Stats sumamry values
        self._LinkEval__aupr = aupr
        self._LinkEval__auroc = auroc
        self.__f1_score = f1_score
        self.__mcc = mcc
        self.__accuracy = accuracy
        self._LinkEval__optimal_cutoff = optimal_cutoff

        # Score metrics distributions
        if inference_id != 'Baseline' or f1_score_dist is not None:
            self._LinkEval__f1_score_dist = f1_score_dist
            self._LinkEval__mcc_dist = mcc_score_dist
            self._LinkEval__accuracy_dist = accuracy_score_dist
            self._LinkEval__score_pre_dist = {score : precision for score, precision in zip(f1_score_dist.keys(), precision_dist[1:-1])}
            self._LinkEval__score_sensi_dist = {score : sensitivity for score, sensitivity in zip(f1_score_dist.keys(), sensitivity_dist[1:-1])}
            self._LinkEval__score_fpr_dist = {score : fpr for score, fpr in zip(f1_score_dist.keys(), fpr_dist[1:-1])}

            if greater_is_better:
                self._LinkEval__cutoff = min(f1_score_dist.keys())
            else:
                self._LinkEval__cutoff = max(f1_score_dist.keys())
            self._LinkEval__repr = f"LinkEval(inference_id={inference_id}, aupr={aupr}, auroc={auroc}, f1_score={f1_score}, mcc={mcc}, accuracy={accuracy}, number_unique_scores={len(f1_score_dist.keys())}, cutoff= {self._LinkEval__cutoff}, optimal_cutoff={optimal_cutoff}, greater_is_better={greater_is_better})"
        else:
            self._LinkEval__f1_score_dist = None
            self._LinkEval__mcc_dist = None
            self._LinkEval__accuracy_dist = None
            self._LinkEval__cutoff = None
            self._LinkEval__score_pre_dist = None
            self._LinkEval__score_sensi_dist = None
            self._LinkEval__score_fpr_dist = None

            self._LinkEval__repr = f"LinkEval(inference_id={inference_id}, aupr={aupr}, auroc={auroc}, f1_score={f1_score}, mcc={mcc}, accuracy={accuracy}, number_unique_scores={None}, cutoff= {self._LinkEval__cutoff}, optimal_cutoff={optimal_cutoff}, greater_is_better={greater_is_better})"

    def __no_computation_warning(self, process:str) -> None:
        stats_logger.warning(f'Computation of {process} unable because input was previously computed benchmarking, not networks')
        return None
    
    def __no_attribute_warning(self, attribute: str) -> None:
        stats_logger.warning(f'Attribute: {attribute} not available when benchmarking stats are provided without networks')
        return None
    
    @property
    def cutoff(self) -> float | None:
        return self._LinkEval__cutoff
    
    @property
    def precision_baseline(self) -> float:
        return self.__no_attribute_warning('precision baseline')

    @property
    def size_negatives(self) -> int:
        return self.__no_attribute_warning('size of negative samples')

    @property
    def size_gold_standard(self) -> int:
        return self.__no_attribute_warning('Gold Standard size')

    @property
    def gold_standard_edges(self) -> set:
        return self.__no_attribute_warning('Gold Standard edges')

    @property
    def inference_edges(self) -> dict[float, set]:
        return self.__no_attribute_warning('inference edges')

    @property
    def size_universe(self) -> int:
        return self.__no_attribute_warning('size of the sample universe')

    @property
    def gold_standard(self) -> str:
        return self.__no_attribute_warning('Gold Standard network')

    @property
    def inference(self) -> str:
        return self.__no_attribute_warning('inference network')

    @property
    def directed(self) -> bool:
        return self.__no_attribute_warning('whether networks benchmarked consider direction')

    @property
    def allow_self_loops(self) -> bool:
        return self.__no_attribute_warning('whether networks benchmarked consider self-loops')

    @cutoff.setter
    def cutoff(self, cutoff: float | None):
        return self.__no_computation_warning('set new cutoff')
    
    def optimal_cutoff_plot(self, ax=None, x_log=False):
        if self._LinkEval__f1_score_dist:
            return super().optimal_cutoff_plot(ax, x_log)
        else:
            return self.__no_computation_warning(f'optimal cutoff plot for {self._LinkEval__inference_id}')
        
    def recall(self, cutoff = None):
        if self._LinkEval__inference_id != 'Baseline':
            return super().recall(cutoff)
        else:
            return self.__no_computation_warning('Baseline recall for specific cutoff')
    
    def precision(self, cutoff = None):
        if self._LinkEval__inference_id != 'Baseline':
            return super().precision(cutoff)
        else:
            return self.__no_computation_warning('Baseline precision for specific cutoff')
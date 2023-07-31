from __future__ import annotations
from typing import Tuple

import gc
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, permutations, product


class NetworkInferenceStats:
    def __init__(
        self,
        gold_standard: str,
        inference: str,
        greater_is_better: bool | None = None,
        directed: bool = True,
        allow_self_loops: bool = False,
    ):
        """
        Class for evaluating binary classification results.
        Optimized for network inference.

        Args:
        gold_standard (str): Path to gold standard file.
            It must be a tab separated file with two columns: source and target.
        inference (str): Path to inference file.
            It must be a tab separated file with two or three columns: source, target and score (optional).
            if score is not provided, the descending order of the edges will be used (better inference first).
        greater_is_better (bool): Whether the inference score is better when it is higher or lower.
            Only used if the inference file has a score column. Otherwise, it is ignored.
            If True, the higher the score, the better the inference.
            If False, the lower the score, the better the inference.
        directed (bool): Whether the network is directed or not.
            If True, the direction of the edges will be considered.
            If False, the direction of the edges will be ignored (i.e., A-B = B-A).

        Notes:
        Node IDs for gold standard and inference must be comparable.
        """

        self.__gold_standard = gold_standard
        self.__inference = inference
        self.__greater_is_better = greater_is_better
        self.__directed = directed
        self.__allow_self_loops = allow_self_loops
        self.__score = False

        with open(inference) as f:
            first_line = f.readline()
            self.__num_columns = len(first_line.split())
        if self.__num_columns not in [2, 3]:
            raise ValueError(
                f"Inference file must have 2 or 3 columns. {self.__num_columns} columns were found."
            )
        elif self.__num_columns == 3:
            self.__score = True
            if self.__greater_is_better is None:
                raise ValueError(
                    "Inference file has 3 columns, but greater_is_better was not provided."
                )
        elif self.__num_columns == 2 and self.__greater_is_better is not None:
            raise ValueError("Inference file has 2 columns, but greater_is_better was provided.")
        elif self.__num_columns == 2:
            # If the inference is not scored, lower index = better inference
            # the order of the edges will be used
            self.__greater_is_better = False

    def __repr__(self) -> str:
        return f"BinClassEval(gold_standard={self.gold_standard}, inference={self.inference}, greater_is_better={self.greater_is_better}, directed={self.directed})"

    def __str__(self) -> str:
        if self._num_columns == 3:
            return f'BinClassEval to evaluate the inference {self.inference} against {self.gold_standard} {"" if self.directed else "not "}\
                considering direction and {"higher" if self.greater_is_better else "lower"} scores are better.'
        else:  # self._num_columns == 2
            return f'BinClassEval to evaluate the inference {self.inference} against {self.gold_standard} {"" if self.directed else "not "}\
                considering direction.'

    def __eq__(self, other: BinClassEval) -> bool:
        raise NotImplementedError

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
    def score(self) -> bool:
        return self.__score

    @property
    def allow_self_loops(self) -> bool:
        return self.__allow_self_loops

    def __read_gold_standard(self, gold_standard_file) -> Tuple[set, set]:
        """Reads the gold standard file and returns a set of edges and a set of genes."""
        gold_standard_edges = set()
        gold_standard_geneset = set()
        with open(gold_standard_file) as f:
            for line in f:
                source, target = line.split()
                gold_standard_edges.add((source, target))
                gold_standard_geneset.add(source)
                gold_standard_geneset.add(target)

        return gold_standard_edges, gold_standard_geneset

    def __read_inference(self, inference_file) -> list:
        """Reads the inference file and returns a set of edges ordered by relevance."""
        inference_edges = defaultdict(list)

        with open(inference_file) as f:
            for i, line in enumerate(f):
                if self.score:
                    source, target, score = line.split()
                    inference_edges[float(score)].append((source, target))
                else:
                    source, target = line.split()
                    # If the inference is not scored, edges order is used
                    # lower index = better inference
                    inference_edges[i].append((source, target))

        # TODO: !!! Ya generalo anonimizado !!!  para ahorrar memoria
        return [
            edges
            for score, edges in sorted(inference_edges.items(), reverse=self.greater_is_better)
        ]

    def __universe(self, gold_standard_geneset: set) -> set:
        """Returns the universe of potential edges between genes in the gold standard.

        The universe in compute following the rules:
        - If the inference is directed and self-loops are allowed, the universe is the cartesian product of the gold standard geneset.
        - If the inference is directed and self-loops are not allowed, the universe is permutations of the gold standard geneset.
        - If the inference is undirected and self-loops are allowed, the universe is combinations with replacement of the gold standard geneset.
        - If the inference is undirected and self-loops are not allowed, the universe is combinations without replacement of the gold standard geneset.
        """
        if self.directed and self.allow_self_loops:
            return set(product(gold_standard_geneset, repeat=2))
        elif self.directed and not self.allow_self_loops:
            return set(permutations(gold_standard_geneset, 2))
        elif not self.directed and self.allow_self_loops:
            return set(combinations_with_replacement(gold_standard_geneset, 2))
        elif not self.directed and not self.allow_self_loops:
            return set(combinations(gold_standard_geneset, 2))

    def __anonymize_edges(self) -> Tuple[set, list[set], int]:
        """Use the universe to anonymize the gold standard and the inference.
        Returns the anonymized gold standard, the anonymized inference and the size of the universe.
        """
        gold_standard_edges, gold_standard_geneset = self.__read_gold_standard(self.gold_standard)
        inference_edges = self.__read_inference(self.inference)
        universe = self.__universe(gold_standard_geneset)
        size_universe = len(universe)

        # Universe is used as reference
        edge_to_id = {edge: i for i, edge in enumerate(universe)}
        gold_standard_edges = {edge_to_id[edge] for edge in gold_standard_edges}
        inference_edges = [{edge_to_id[edge] for edge in edges} for edges in inference_edges]

        # erase universe
        del universe
        del edge_to_id
        # call garbage collector
        gc.collect()

        return gold_standard_edges, inference_edges, size_universe


# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, 'foo.tsv')

# foo = BinClassEval('a', file_path)
# print(foo.greater_is_better)
# print(foo)

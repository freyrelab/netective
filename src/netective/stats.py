from __future__ import annotations


class BinClassEval:
    def __init__(
        self,
        gold_standard: str,
        inference: str,
        greater_is_better: bool | None = None,
        directed: bool = True,
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
        self.__score = False

        with open(inference) as f:
            first_line = f.readline()
            self.__num_columns = len(first_line.split())
        if self.__num_columns not in [2, 3]:
            raise ValueError(
                f"Inference file must have 2 or 3 columns. {self.__num_columns} columns were found."
            )
        if self.__num_columns == 3:
            self.__score = True
            if self.__greater_is_better is None:
                raise ValueError(
                    "Inference file has 3 columns, but greater_is_better was not provided."
                )
        if self.__num_columns == 2 and self.__greater_is_better is not None:
            raise ValueError(
                "Inference file has 2 columns, but greater_is_better was provided."
            )

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

    def universe(self) -> set:
        ...


# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, 'foo.tsv')

# foo = BinClassEval('a', file_path)
# print(foo.greater_is_better)
# print(foo)

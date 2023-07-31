from __future__ import annotations

import os
import math
import uuid
import numpy as np
import pandas as pd
import hashlib
from warnings import warn
from networkx import Graph
from networkx import DiGraph

from freyrelab.regnets import regnet as rn
from freyrelab.nets import powerlaw as pl

from netective.utils import *

import matplotlib.pyplot as plt


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

    def __init__(self, G: Graph | rn.RegNet, data: bool = False) -> None:
        """
        Initialize the GraphObserver class.

        Args:
            G (nx.Graph | rn.RegNet): The graph to observe.
            data (bool): If True, the node/edge data will be considered when computing the hash.

        """
        self.G = G
        self.data = data
        self.graph_hash = self._compute_hash(self.G)

    def _compute_hash(self, G: Graph | rn.RegNet) -> str:
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

    def changed(
        self, G: Graph | rn.RegNet = None, update_G: bool = False
    ) -> bool:
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


class Structure(pd.Series):
    def __init__(
        self,
        G: DiGraph | rn.RegNet,
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

        super().__init__()  # DataFrame.__init__(self)
        print(net_id)
        self.G = validate_network(
            G
        )  # network to compute the structural properties
        self.graph_observer = GraphObserver(G)
        self.graph_observer.graph_hash = None  # hash of the network to detect changes. None means that the properties have not been computed yet.
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

    def get_props(self) -> pd.Series:
        """
        Computes the structural properties of a network.

        Returns:
            pd.Series: Series with the structural properties of the network.
        """
        # Compute the properties if they have not been computed yet or if the network has changed
        if (
            self.graph_observer.graph_hash is None
            or self.graph_observer.changed(self.G, update_G=True)
        ) or self.norm_observer.change():
            self._props = self.compute_props()

        return self._props

    def _fit_powerlaw_ck(self, CK: tuple[int, float]) -> float | np.nan:
        """Fits a power law to the clustering coefficient distribution of a network."""
        try:
            prop_ck = CK.rsquared_adj
        except ValueError:
            warn(
                f"Clusterings for {self.net_id} cannot be fitted to a power law."
            )
            prop_ck = np.nan
        return prop_ck

    def _fit_powerlaw_pk(self, k: list[int]) -> float | np.nan:
        """Fits a power law to the degree distribution of a network."""
        try:
            PK = pl.Pk(k)
            prop_pk = PK.rsquared_adj
        except ValueError:
            warn(f"Degrees for {self.net_id} cannot be fitted to a power law.")
            prop_pk = np.nan
        return prop_pk

    def _kappa(self, CK: tuple[int, float]) -> float | np.nan:
        """Computes the kappa coefficient of a network."""
        try:
            if not np.isnan(CK.kappa):
                props_kappa = round(CK.kappa)
            else:
                props_kappa = np.nan
        except ValueError:
            print(self.net_id)
            warn(
                f"Kappa for {self.net_id} cannot be calculated due to its non-hierarchical structure."
            )
            props_kappa = np.nan
        return props_kappa

    def _norm_props(self, props: pd.Series) -> pd.Series:
        """Normalizes the structural properties of a network."""

        if self.verbose:
            print("Normalizing...", flush=True)

        available_norms = ["biol"]
        if (
            not isinstance(self.norm_observer.norm, pd.Series)
            and self.norm_observer.norm not in available_norms
        ):
            raise ValueError(
                f"Normalization factor {self.norm_observer.norm} not recognized."
            )

        if self.norm_observer.norm == "biol":
            self.norm_observer.norm = self._bio_scale(props)

        # else is a pd.Series with the normalization factors
        props = props.multiply(
            self.norm_observer.norm, fill_value=np.nan
        )  # missing values are reported as NaN (non desired properties)

        return props

    def compute_props(self) -> dict[str, float, int]:

        """
        Computes the structural properties of a network.

        Args:
            G: DiGraph or RegNet.
                Network to compute the structural properties.
            norm: None|str|pd.Series.
                Normalization factor for each property.
            net_id: str.
                Name of the network. If None, a random uuid is assigned.

        Returns:
            dict: Dictionary with the structural properties of the network.
        """

        if self.verbose:
            print(f"Processing {self.net_id}...", flush=True)
            print(
                f"{self.net_id} has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.",
                flush=True,
            )

        self.G.remove_isolates()
        props = {}

        props["Density"] = self.G.density
        props["Regulators"] = self.G.regulators_count
        props["Self regulations"] = self.G.selfinteractions_count
        props["Max. out connectivity"] = self.G.kout_max

        # in-place
        self.G.remove_selfinteractions()
        self.G.remove_isolates()

        # Props without selfloops
        props["3-Feedback loops"] = self.G.feedbacks3_count
        props["Feedforward circuits"] = self.G.feedforwards_count
        props[
            "Complex feedforward circuits"
        ] = self.G.complex_feedforwards_count
        props["Genes in the giant component"] = self.G.giant_component_size

        props["Diameter"] = self.G.diameter()
        props["Average shortest path length"] = self.G.average_path_length()
        props[
            "Average clustering coefficient"
        ] = self.G.average_clustering_coefficient

        # C(k) and P(k)
        kc = self.G.k_clustering()  # {node: (k, c)}

        CK = pl.Ck(kc.values())
        props["R^2 C(k)"] = self._fit_powerlaw_ck(CK)

        k, _ = zip(*kc.values())
        props["R^2 P(k)"] = self._fit_powerlaw_pk(k)

        # Kappa value
        kc = self.G.k_clustering(kdir="out")
        CK = pl.Ck(kc.values())
        props["Kappa"] = self._kappa(CK)

        # Convert to pd.Series
        props = pd.Series(props)

        # Normalization
        if self.norm_observer.norm is not None:
            props = self._norm_props(props)

        return props

    def _bio_scale(self, props: pd.Series) -> pd.Series:

        """
        USe the biological maximum theoretical values as scale factors."

        Args:
            props: pd.Series.
                Structural properties of the network.
            G: DiGraph or RegNet.
                Network to compute the structural properties.

        Returns:
            pd.Series: Scale factors for each property.
        """

        warn("Normalization for clustering coefficient not implemented yet")

        n_genes = self.G.number_of_nodes()
        tfs = props["Regulators"]
        max_3tfs_loop = _max_loops(n=n_genes, r=3, tfs=tfs, r_tfs=3)
        max_2tfs_loop = _max_loops(n=n_genes, r=3, tfs=tfs, r_tfs=2)
        largest_putative_path = props["Genes in the giant component"] - 1

        scalling_f = {
            "Density": n_genes
            / tfs,  # equivalent to E / (n**2 * (tfs/n)) # TODO: corrects previous mistake: G.density * (G.regulators_count / nGenes) << 1
            "Regulators": 1 / n_genes,  # fraction of nodes that are regulators
            "Self regulations": 1
            / tfs,  # fraction of regulators that self-regulate
            "Max. out connectivity": 1
            / n_genes,  # fraction of nodes that are regulated by the most hub
            "3-Feedback loops": 1
            / max_3tfs_loop,  # fraction of possible 3-feedback loops
            "Feedforward circuits": 1
            / max_2tfs_loop,  # fraction of possible feedforward circuits
            "Complex feedforward circuits": 1
            / max_2tfs_loop,  # fraction of possible complex feedforward circuits A->B->C, A->C, B->A
            "Genes in the giant component": 1
            / n_genes,  # fraction of nodes in the giant component
            "Diameter": 1
            / largest_putative_path,  # denominator is equivalent to (n-2+1) for the nodes in the giant component. n-2 (excluding sorce and target nodes) + 1 (we are coiunting edges). # TODO: corrects previous mistake: props['Diameter'] / (n_genes-2)
            "Average shortest path length": 1
            / largest_putative_path,  # denominator is equivalent to (n-2+1) for the nodes in the giant component. # TODO corrects previous mistake: / (n_genes-2)
            "Average clustering coefficient": 1,  # TODO: This still needs to be normalized
            "R^2 C(k)": 1,  # already normalized
            "R^2 P(k)": 1,  # already normalized
            "Kappa": 1
            / n_genes,  # fraction of nodes that are regulated by the most hub
        }

        return pd.Series(scalling_f)


def struc_props_call(
    G: DiGraph | rn.RegNet,
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
            ER = rn.RegNet(
                fast_gnp_random_graph(n, m / (n**2), directed=True)
            )
            S_er = Structure(
                ER, norm=norm, net_id=f"{net_id}_ER_{i}", verbose=verbose
            )
            props_i = S_er.get_props()
            for k, v in props_i.items():
                props_er[k].append(v)

        # TODO: here you can make it robust to a threshold of nan values
        props_er_avg = {
            prop: sum(vals) / len(vals) for prop, vals in props_er.items()
        }
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

    file_p_s = concat_path(output, 'network_level_comp.png')
    file_p_d = concat_path(output, 'node_level_comp.png')

    # save output
    # rewrite file if it exists
    # with open(file_p, "w") as f:
        # save command line
        # print(cl, file=f)

    # save structural properties
    fig_scalar.savefig(file_p_s, dpi=300)
    fig_dist.savefig(file_p_d, dpi=300)

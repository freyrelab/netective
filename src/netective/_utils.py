from __future__ import annotations
"""Utility functions for the netective package."""
__all__ = ['concat_path', 'newline']

import os
import networkx as nx

concat_path = os.path.join

def parse_nets(paths: list[str], comments: str='#', delimiter: str='\t') -> dict:

    """Reads network files and returns a dictionary of networkx.DiGraphs.

    Firts column of the network file is considered as the source node and the
    second column is considered as the target node. The network file must be
    delimited by a tab character.
    
    Args:
        paths (list[str]): List of paths to network files.
        comments (str, optional): Comment character. Defaults to '#'.
        delimiter (str, optional): Delimiter character. Defaults to '\t'.

    Returns:
        dict: Dictionary of networkx.DiGraphs.

    Raises:
        ValueError: If the network file is not a DiGraph.

    TODO:
        * Add support for metadata (scores of the predictions).
        * Raise Error when len(tfs & tgs) = 0.
    """

    networks = {}

    for net_path in paths:
                    
        net_name = os.path.basename(net_path)

        # read network file (only DiGraphs with no metadata are supported)
        networks[net_name] = nx.read_edgelist(
            net_path,
            comments=comments,
            delimiter=delimiter,
            create_using=nx.DiGraph,
            data=False,
            encoding='utf-8'
            )

    return networks
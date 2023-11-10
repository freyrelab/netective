import os
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import parse_network, save_prop_dicts, save_figs, common_props_dict, association, sort_files
from netective.structure.dataviz import create_symmetric_heatmap
from netective import compare_structure, characterize_network

import networkx as nx

from netective.logging_info import get_logger, set_log_level

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass

cli_logger = get_logger(__name__)

def main():
    ## parse arguments
    args = _parse_arguments()
    print(args)

if __name__ == "__main__":
    
    main()
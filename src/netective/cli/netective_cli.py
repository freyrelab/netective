import os
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import parse_network, save_prop_dicts, save_figs, common_props_dict, association, sort_files, get_clusters
from netective.structure.dataviz import create_symmetric_heatmap
from netective import compare_structure, characterize_network

import networkx as nx
import pandas as pd

from netective.logging_info import get_logger, set_log_level

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass

cli_logger = get_logger(__name__)

def runmode1(args):
    pass

def runmode2(args):
    # Args for network analysis
    nets_path = args.input
    norm = args.normalization
    verbose = args.verbose
    if verbose != None:
        set_log_level(cli_logger, verbose)
    if len(args.selected_props) != 1:
        selected_props = args.selected_props
    elif args.selected_props[0] == 'all':
        selected_props = 'all'
    else:
        selected_props = args.selected_props
    if not args.workers:
        workers = cpu_count() - 1
        cli_logger.warning(f'Multiprocessing enabled in {workers} usable threads detected.')
    else:
        workers = cpu_count() - 1 if args.workers > cpu_count() - 1 else args.workers
        cli_logger.warning(f'Multiprocessing enabled in {workers} threads.')
    return_prop_dicts = args.keep_props
    erdos_renyi = args.erdos_renyi

    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments} command: python {__file__} --path {nets_path} --norm {norm} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {delimiter} --output {output}\n"

    scalars_array = {}
    dist_array = {}
    networks = {}
    # Network analysis
    sorted_files = sort_files(path= nets_path)
    complete_batches = len(sorted_files) // workers
    last_batch = len(sorted_files) % workers
    completed = 0
    
    if len(sorted_files) < 2:
        cli_logger.critical('Only one network detected in inputed directory. For network characterization enter full path to network, not path to directory.')
        exit ()
    for net_path in sorted_files:
        net_id = os.path.basename(net_path)
        networks[net_id] = parse_network(net_path, comments, delimiter)

        # Number of inputed nets is > to workers, batch processing
        if len(sorted_files) > workers and (len(networks) == workers or (len(networks) == last_batch and completed == complete_batches)):
            temp_scalars_array, temp_dist_array = compare_structure(
                networks= networks, 
                norm= norm, 
                workers= workers, 
                selected_props= selected_props,
                verbose= verbose,
                return_prop_dicts= True,
                erdos_renyi= erdos_renyi,
                process= f'analysis of INPUTED networks: {list(networks.keys())}'
            )
            scalars_array.update(temp_scalars_array)
            dist_array.update(temp_dist_array)
            networks = {}
            completed += 1
    # Number of inputed nets is <= to workers
    if len(sorted_files) <= workers:
        scalars_array, dist_array = compare_structure(
            networks= networks, 
            norm= norm, 
            workers= workers, 
            selected_props= selected_props,
            verbose= verbose,
            return_prop_dicts= True,
            erdos_renyi= erdos_renyi,
            process= f'analysis of INPUTED networks: {list(networks.keys())}'
        )
    if return_prop_dicts:
        if args.runmode == 4:
            return scalars_array
        for net_id, props in scalars_array.items():
            save_prop_dicts(
                array= props,
                net_id= net_id,
                type= 'scalars',
                output_dir= output,
                delimiter= delimiter,
                cl= cl
            )
        for net_id, props in dist_array.items():
            save_prop_dicts(
                array= props,
                net_id= net_id,
                type= 'distributions',
                output_dir= output,
                delimiter= delimiter,
                cl= cl
            )
    else:
        scalars_array = common_props_dict(scalars_array)
        # Scalar properties
        if len(scalars_array) > 0 and len(list(scalars_array.values())[0]) > 1:
            df = association(scalars_array)
            fig_scalars = create_symmetric_heatmap(df, title=f"Global properties", verbose= verbose)
            save_figs(fig_scalars, output_dir= output)
        else:
            cli_logger.critical("Not enough data to compare.")
            raise ValueError("Not enough data to compare.")
    

def runmode3(args):
    pass

def runmode4(args):
    scalars = runmode2(args)
    merged_df = pd.DataFrame.from_dict(scalars).T
    merged_df.dropna(axis=1, inplace=True, how='any')
    clusters = get_clusters(merged_df.T.corr(), map_ids= True)
    print(clusters)

def main():
    ## parse arguments
    args = _parse_arguments()
    if args.runmode == 1:
        pass
    elif args.runmode == 2:
        runmode2(args)
    elif args.runmode == 3:
        pass
    elif args.runmode == 4:
        runmode4(args)

if __name__ == "__main__":
    
    main()
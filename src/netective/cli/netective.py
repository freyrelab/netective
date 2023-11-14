import os
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import parse_network, save_prop_dicts, save_figs, common_props_dict, association, sort_files, get_clusters
from netective.structure.dataviz import create_symmetric_heatmap, plot_distributions, plot_scalars
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
RUNMODES = {
    1 : 'Characterize',
    2 : 'Compare',
    3 : 'Assess',
    4 : 'Classify'
}

def runmode1(args):
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
    
    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments} command: python {__file__} {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"
    
    scalars_array, dist_array = compare_structure(
                networks= nets_path, 
                norm= norm, 
                workers= workers, 
                selected_props= selected_props,
                verbose= verbose,
                return_prop_dicts= True
    )
    
    if return_prop_dicts:
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
        for net_id, props in scalars_array.items():
            fig_scalar, _ = plot_scalars(props, verbose= verbose)
            save_figs(
                fig= fig_scalar,
                props= 'scalars',
                net_id= net_id,
                compare= False,
                output_dir= output
            )
        for net_id, props in dist_array.items():
            fig_dist, _ = plot_distributions(props, verbose= verbose)
            save_figs(
                fig= fig_dist,
                props= 'distributions',
                net_id= net_id,
                compare= False,
                output_dir= output
            )

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

    cl = f"{comments} command: python {__file__} {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"

    scalars_array, dist_array = compare_structure(
        networks= nets_path, 
        norm= norm, 
        workers= workers, 
        selected_props= selected_props,
        verbose= verbose,
        return_prop_dicts= True,
        erdos_renyi= erdos_renyi,
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
    # Structural comparison
    scalars = runmode2(args)
    cli_logger.warning('Starting classification of networks into clusters...')
    merged_df = pd.DataFrame.from_dict(scalars).T
    merged_df.dropna(axis=1, inplace=True, how='any')
    method = args.method
    metric = args.metric
    clusters_num = args.clusters
    threshold = args.threshold if clusters_num is None else None
    cl = f"\n{args.comments} command: python {__file__} {RUNMODES[args.runmode]} --path {args.input} --norm {args.normalization} --clusters {clusters_num} --threshold {threshold} --method {method} --metric {metric} --workers {args.workers} --verbose {args.verbose} --comments {args.comments} --delimiter {repr(args.delimiter)} --output {args.output}"
    clusters = get_clusters(
        merged_df.T.corr(),
        map_ids= True,
        ch_method= method,
        ch_metric= metric,
        clust_num= clusters_num,
        threshold= threshold
    )

    for clust, nets in clusters.items():
        print(f'Cluster {clust}: {nets}')

def main():
    ## parse arguments
    args = _parse_arguments()
    if args.runmode == 1:
        runmode1(args)
    elif args.runmode == 2:
        runmode2(args)
    elif args.runmode == 3:
        runmode3(args)
    elif args.runmode == 4:
        runmode4(args)

if __name__ == "__main__":
    main()
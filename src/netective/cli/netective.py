import os
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import save_prop_dicts, save_figs, common_props_dict, association, get_clusters, parse_network
from netective.structure.dataviz import create_symmetric_heatmap, plot_distributions, plot_scalars
from netective import compare_structure, characterize_network

import pandas as pd

from netective.logging_info import get_logger, set_log_level

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass

cli_logger = get_logger(__name__)
RUNMODES = {
    1 : 'characterize',
    2 : 'compare',
    3 : 'assess',
    4 : 'classify'
}
concat_path = os.path.join

def runmode1(args):
    # Args for network analysis
    nets_path = args.input
    norm = args.normalization
    
    verbose = args.verbose
    if verbose is not None:
        set_log_level(cli_logger, verbose)
    
    if len(args.selected_props) != 1:
        selected_props = args.selected_props
    elif args.selected_props[0] == 'all':
        selected_props = 'all'
    else:
        selected_props = args.selected_props
    
    usable_threads = cpu_count() - 1
    if args.workers == 'auto':
        workers = usable_threads
    else:
        try:
            workers = int(args.workers)
            workers = usable_threads if workers > usable_threads else workers
        except:
            workers = usable_threads
    cli_logger.warning(f'Multiprocessing enabled in {workers} out of {usable_threads} usable threads detected')
    
    return_prop_dicts = args.keep_props
    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"
    
    if os.path.isdir(nets_path):
        scalars_array, dist_array = compare_structure(
                    networks= nets_path, 
                    norm= norm, 
                    workers= workers, 
                    selected_props= selected_props,
                    verbose= verbose,
                    return_prop_dicts= True,
                    keep_averages= False
        )
    else:
        net_id = os.path.basename(nets_path)
        net = parse_network(nets_path, comments, delimiter)
        cli_logger.warning('No parallelization required, only one network inputed')
        scalars_array, dist_array = characterize_network(
            G= net,
            net_id= net_id,
            norm= norm,
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
    if verbose is not None:
        set_log_level(cli_logger, verbose)
    
    if len(args.selected_props) != 1:
        selected_props = args.selected_props
    elif args.selected_props[0] == 'all':
        selected_props = 'all'
    else:
        selected_props = args.selected_props
    
    usable_threads = cpu_count() - 1
    if args.workers == 'auto':
        workers = usable_threads
    else:
        try:
            workers = int(args.workers)
            workers = usable_threads if workers > usable_threads else workers
        except:
            workers = usable_threads
    cli_logger.warning(f'Multiprocessing enabled in {workers} out of {usable_threads} usable threads detected')
    
    return_prop_dicts = args.keep_props
    erdos_renyi = args.erdos_renyi

    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"

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
    method = args.method
    metric = args.metric
    clusters_num = args.clusters
    output = args.output
    threshold = args.threshold if clusters_num is None else None
    cl = f"{args.comments}command: netective {RUNMODES[args.runmode]} --path {args.input} --norm {args.normalization} --clusters {clusters_num} --threshold {threshold} --method {method} --metric {metric} --workers {args.workers} --verbose {args.verbose} --comments {args.comments} --delimiter {repr(args.delimiter)} --output {output}"
    # Structural comparison
    scalars = runmode2(args)
    cli_logger.info('Starting classification of networks into clusters...')
    merged_df = pd.DataFrame.from_dict(scalars).T
    merged_df.dropna(axis=1, inplace=True, how='any')
    clusters = get_clusters(
        merged_df.T.corr(),
        map_ids= True,
        ch_method= method,
        ch_metric= metric,
        clust_num= clusters_num,
        threshold= threshold
    )
    if output is not None:
            if not os.path.isdir(output):
                cli_logger.warning(f'Invalid output {output}, setting current directory instead')
                output = os.getcwd()
            exts = {",": "csv", "\t": "tsv"}
            ext = exts.get(args.delimiter, "txt")
            output_file = concat_path(output, f"networks_classification.{ext}")
            f = open(output_file, 'w')
            f.write(f'{cl}\n{args.comments}Network{args.delimiter}Cluster')
    else:
        print(f'Network{args.delimiter}Cluster')
    
    for clust, nets in clusters.items():
        for net in nets:
            if output is not None:
                f.write(f'\n{net}{args.delimiter}{clust}')
            else:
                print(f'{net}{args.delimiter}{clust}')
    try:
        f.close()
    except UnboundLocalError:
        pass

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
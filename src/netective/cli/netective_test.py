import os
from multiprocessing import cpu_count

from netective.cli import _arguments
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
    args = _arguments._parse_arguments()

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
    
    # Creation of dictionary with networks
    networks = {}
    if not os.path.isdir(nets_path):
        networks[os.path.basename(nets_path)] = parse_network(nets_path, comments, delimiter)
        net_id = list(networks.keys())[0]
        network = list(networks.values())[0]
        if return_prop_dicts:
            scalars_array, dist_array = characterize_network(
                G= network,
                net_id= net_id,
                norm= norm,
                selected_props= selected_props,
                verbose= verbose,
                return_prop_dicts= True
            )
            save_prop_dicts(
                array= scalars_array,
                net_id= net_id,
                type= 'scalars',
                output_dir= output,
                delimiter= delimiter,
                cl= cl
            )
            save_prop_dicts(
                array= dist_array,
                net_id= net_id,
                type= 'distributions',
                output_dir= output,
                delimiter= delimiter,
                cl= cl
            )
        else:
            fig_scalars, fig_dist = characterize_network(
                G= network,
                net_id = net_id,
                selected_props= selected_props,
                return_prop_dicts= False,
                verbose= verbose
            )
            save_figs(
                fig= fig_scalars,
                props= 'scalars',
                net_id= net_id,
                compare= False,
                output_dir= output
            )
            save_figs(
                fig= fig_dist,
                props= 'distributions',
                net_id= net_id,
                compare= False,
                output_dir= output
            )
    
    # More than one network
    else:
        scalars_array = {}
        dist_array = {}
        completed = 0
        # Network analysis
        sorted_files = sort_files(path= nets_path)
        
        for net_path in sorted_files:
            net_id = os.path.basename(net_path)
            networks[net_id] = parse_network(net_path, comments, delimiter)

            # Number of inputed nets is > to workers, batch processing
            if (len(networks) == workers or completed == len(sorted_files) // workers) and len(sorted_files) > workers:
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

        # 
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
            scalars_array = common_props_dict(scalars_array)
            # Scalar properties
            if len(scalars_array) > 0 and len(list(scalars_array.values())[0]) > 1:
                df = association(scalars_array)
                fig_scalars = create_symmetric_heatmap(df, title=f"Global properties", verbose= verbose)
                save_figs(fig_scalars, output_dir= output)
            else:
                cli_logger.critical("Not enough data to compare.")
                raise ValueError("Not enough data to compare.")

if __name__ == "__main__":
    main()                                                                                                                   

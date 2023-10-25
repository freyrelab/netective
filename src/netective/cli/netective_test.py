import os
from pandas import DataFrame

from netective.cli import _arguments
from netective.utils import parse_network, save_prop_dicts, save_figs

from netective import compare_structure, characterize_network

from matplotlib.pyplot import savefig
import networkx as nx

try:
    import pretty_traceback

    pretty_traceback.install()
except ImportError:
    pass


def main():

    ## parse arguments
    args = _arguments._parse_arguments()

    # Args for network analysis
    nets_path = args.path
    norm = args.norm
    if len(args.props) != 1:
        selected_props = args.props
    elif args.props[0] == 'all':
        selected_props = 'all'
    else:
        selected_props = args.props
    try:
        workers = int(args.workers)
    except ValueError:
        workers = args.workers
    return_prop_dicts = args.keep_props
    verbose = args.verbose
    erdos_renyi = args.erdos_renyi

    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output
    output_file = args.output_file

    cl = f"{comments} command: python {__file__} --path {nets_path} --norm {norm} --comments {comments} --delimiter {delimiter} --output {output} --output_file {output_file} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose}"
    
    # Creation of dictionary with networks
    networks = {}
    # print(os.path.isdir(nets_path))
    for root, dir, files in os.walk(nets_path):
        if len(files) != 0:
            for net_name in files:
                net_path = os.path.join(os.getcwd(), root, net_name)
                networks[net_name] = parse_network(net_path, comments, delimiter)
    

    if len(networks) > 1:
    # Multiple networks to compare
        if return_prop_dicts:
            scalars_array, dist_array = compare_structure(
                networks= networks, 
                norm= norm, 
                workers= workers, 
                selected_props= selected_props,
                verbose= verbose,
                return_prop_dicts= True,
                erdos_renyi= erdos_renyi
            )
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
            fig_scalars = compare_structure(
                networks=networks, 
                norm= norm, 
                workers= workers, 
                selected_props= selected_props,
                verbose= verbose,
                erdos_renyi= erdos_renyi,
                return_prop_dicts= False
            )
            # Aqui va la fxn que guarde la fig
            save_figs(fig_scalars, output_dir= output, cl= cl)
    
    # Only one network to characterize
    else:
        net_id = list(networks.keys())[0]
        network = list(networks.values())[0]
        if return_prop_dicts:
            scalars_array, dist_array = characterize_network(
                G= network,
                net_id= net_id,
                norm= norm,
                selected_props= selected_props,
                verbose= verbose,
                return_prop_dicts= return_prop_dicts
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
            # Aqui va la fxn que guarde las figs
            save_figs(
                fig= fig_scalars,
                type= 'scalars',
                net_id= net_id,
                compare= False,
                output_dir= output,
                cl= cl
            )
            save_figs(
                fig= fig_dist,
                type= 'distributions',
                net_id= net_id,
                compare= False,
                output_dir= output,
                cl= cl
            )


## read arguments
if __name__ == "__main__":
    main()


# H:\\Mi unidad\\Respaldo\\Genomicas\\netective\\data\\test_comp
# H:\\Mi unidad\\Respaldo\\Genomicas\\netective\\data\\test_charac
# python netective_test.py --path ..\..\data\test --norm biological                                                                                                                     

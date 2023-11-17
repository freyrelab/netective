import os
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import save_prop_dicts, save_figs, common_props_dict, association, get_clusters, parse_network
from netective.structure.dataviz import create_symmetric_heatmap, plot_distributions, plot_scalars
from netective import compare_structure, characterize_network, benchmarking

import pandas as pd
import matplotlib.pyplot as plt

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
    dir = args.directed
    
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
    workers = args.workers
    
    return_prop_dicts = args.keep_props
    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --dir {dir} --workers {workers} --selected_props {selected_props} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"
    
    if os.path.isdir(nets_path):
        cli_logger.warning(f'Multiprocessing enabled in {workers} out of {usable_threads} usable threads detected')
        scalars_array, dist_array = compare_structure(
                    networks= nets_path, 
                    norm= norm, 
                    workers= workers, 
                    selected_props= selected_props,
                    verbose= verbose,
                    return_prop_dicts= True,
                    keep_averages= False,
                    directed= dir,
                    delimiter= delimiter,
                    comments= comments
        )
    else:
        net_id = os.path.basename(nets_path)
        net = parse_network(
                file_path= nets_path,
                comments= comments,
                delimiter= delimiter,
                directed= dir,
                use_position_as_score= False
        )
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
    dir = args.directed
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
    workers = args.workers

    cli_logger.warning(f'Multiprocessing enabled in {workers} out of {usable_threads} usable threads detected')
    
    return_prop_dicts = args.keep_props
    erdos_renyi = args.erdos_renyi

    # Technical Args
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --directed {dir} --selected_props {selected_props} --erdos_renyi {erdos_renyi} --workers {workers} --verbose {verbose} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"

    scalars_array, dist_array = compare_structure(
        networks= nets_path, 
        norm= norm, 
        workers= workers, 
        selected_props= selected_props,
        verbose= verbose,
        return_prop_dicts= True,
        erdos_renyi= erdos_renyi,
        directed= dir,
        delimiter= delimiter,
        comments= comments
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
            cli_logger.critical("Not enough data to compare. Probably input directory has only one network file.")
            raise ValueError("Not enough data to compare.")

def runmode3(args):
    gs = args.gold_standard
    inferences = args.inferences
    directed = args.directed
    greater_is_better = args.greater_is_better
    keep_auc_dicts = args.keep_auc_dicts
    cutoff = args.cutoff
    score = args.score
    self_loops = args.self_loops
    verbose = args.verbose
    output = args.output
    comments = args.comments
    delimiter = args.delimiter
    cl = f'{comments}command: netective {RUNMODES[args.runmode]} --gold_standard {gs} --inferences {inferences} --directed {directed} --greater_is_better {greater_is_better} --keep_auc_dicts {keep_auc_dicts} --cutoff {cutoff} --score {score} --self_loops {self_loops} --verbose {verbose} --output {output} --delimiter {repr(delimiter)} --comments {comments}'
    if keep_auc_dicts: # Keeping dictionaries with aupr and auroc values
        aupr_scores, auroc_scores = benchmarking(
            networks= inferences,
            gold_standard= gs,
            directed= directed,
            greater_score_is_better= greater_is_better,
            allow_self_loops= self_loops,
            cutoff= cutoff,
            return_auc_dicts= True,
            comments= comments,
            delimiter= delimiter,
            score= score,
            verbose= verbose
        )
        if output is not None:
            if not os.path.isdir(output):
                cli_logger.warning(f'Invalid output {output}, setting current directory instead')
                output = os.getcwd()
            exts = {",": "csv", "\t": "tsv"}
            ext = exts.get(args.delimiter, "txt")
            aupr_output_file = concat_path(output, f"aupr_scores.{ext}")
            auroc_output_file = concat_path(output, f"auroc_scores.{ext}")
            aupr_f = open(aupr_output_file, 'w')
            auroc_f = open(auroc_output_file, 'w')
            aupr_f.write(f'{cl}\n{comments}AUPR Scores\n{comments}Network{delimiter}Score')
            auroc_f.write(f'{cl}\n{comments}AUROC Scores\n{comments}Network{delimiter}Score')
        else:
            print(f'\t\tAUPR Scores\nNetwork{args.delimiter}Score')
        
        for net, value in aupr_scores.items():
            if output is not None:
                aupr_f.write(f'\n{net}{delimiter}{value}')
            else:
                print(f'{net}{delimiter}{value}')
        for i,(net, value) in enumerate(auroc_scores.items()):
            if output is not None:
                auroc_f.write(f'\n{net}{delimiter}{value}')
            else:
                if i == 0:
                    print(f'\n\t\tAUROC Scores\nNetwork{delimiter}Score')
                print(f'{net}{delimiter}{value}')
        try:
            aupr_f.close()
            auroc_f.close()
        except UnboundLocalError:
            pass
      
    else:
        fig_aupr, fig_pr_curves, fig_auroc, fig_roc_curves = benchmarking(
            networks= inferences,
            gold_standard= gs,
            directed= directed,
            greater_score_is_better= greater_is_better,
            allow_self_loops= self_loops,
            cutoff= cutoff,
            return_auc_dicts= False,
            comments= comments,
            delimiter= delimiter,
            score= score,
            verbose= verbose
        )
        if not os.path.isdir(output):
            cli_logger.warning(f'No output directory stated, setting current directory as output directory.')
            output = os.getcwd()
        fig_aupr.get_figure().savefig(fname= concat_path(output, f'aupr.png'), bbox_inches= 'tight', dpi= 300)
        fig_pr_curves.get_figure().savefig(fname= concat_path(output, f'pr_curves.png'), bbox_inches= 'tight', dpi= 300)
        fig_auroc.get_figure().savefig(fname= concat_path(output, f'auroc.png'), bbox_inches= 'tight', dpi= 300)
        fig_roc_curves.get_figure().savefig(fname= concat_path(output, f'roc_curves.png'), bbox_inches= 'tight', dpi= 300)

def runmode4(args):
    dir = args.directed
    method = args.method
    metric = args.metric
    clusters_num = args.clusters
    output = args.output
    threshold = args.threshold if clusters_num is None else None
    cl = f"{args.comments}command: netective {RUNMODES[args.runmode]} --path {args.input} --norm {args.normalization} --directed {dir} --selected_props {args.selected_props} --clusters {clusters_num} --threshold {threshold} --method {method} --metric {metric} --workers {args.workers} --verbose {args.verbose} --comments {args.comments} --delimiter {repr(args.delimiter)} --output {output}"
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
import os
import numpy as np
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import save_prop_dicts, save_figs, common_props_dict, association, get_clusters, parse_network, clean_names_association_df, get_models_abbreviations, filter_association_df_for_models
from netective.structure.dataviz import create_comp_heatmap, plot_distributions, plot_scalars
from netective import compare_structure, characterize_network, benchmarking
from netective.structure.structure import __get_optimal_workers

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
    nets_file_format = args.net_f_format
    comments = args.comments
    delimiter = args.delimiter
    output = args.output

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --dir {dir} --workers {workers} --selected_props {selected_props} --verbose {verbose} --nets_f_format {nets_file_format} --comments {comments} --delimiter {repr(delimiter)} --output {output}\n"

    if os.path.isdir(nets_path):
        if workers == 'auto':
            cli_logger.warning('Getting optimal number of workers based on available memory and inputed networks sizes...')
            workers = __get_optimal_workers(
                nets = nets_path,
                directed= dir,
                comments= comments,
                delimiter= delimiter,
                nets_file_format= nets_file_format
            )
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
                    nets_file_format= nets_file_format,
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
                use_position_as_score= False,
                net_file_format= nets_file_format
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
    exts = {",": "csv", "\t": "tsv"}
    # Args for network analysis
    nets_path = args.input
    norm = args.normalization
    dir = args.directed
    verbose = args.verbose
    association_metric = args.association
    metric = args.metric
    method = args.method
    compare_to_models = args.comp2models
    n_random_models = args.n_models
    ba_m = args.m4ba
    keep_averages = args.keep_averages

    if verbose is not None:
        set_log_level(cli_logger, verbose)
    
    if len(args.selected_props) != 1:
        selected_props = args.selected_props
    elif args.selected_props[0] == 'all':
        selected_props = 'all'
    else:
        selected_props = args.selected_props
    
    workers = args.workers
    
    return_prop_dicts = args.keep_props

    # Technical Args
    nets_file_format = args.net_f_format
    comments = args.comments
    delimiter = args.delimiter
    output = args.output
    title = args.title

    if compare_to_models:
        cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --directed {dir} --selected_props {selected_props} --association {association_metric} --metric {metric} --method {method} --comp2models {compare_to_models} --n_models {n_random_models} --m4ba {ba_m} --workers {workers} --verbose {verbose} --net_f_format {nets_file_format} --comments {comments} --delimiter {repr(delimiter)} --title {title} --output {output}\n"
    else:
        cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --directed {dir} --selected_props {selected_props} --association {association_metric} --metric {metric} --method {method} --workers {workers} --verbose {verbose} --net_f_format {nets_file_format} --comments {comments} --delimiter {repr(delimiter)} --title {title} --output {output}\n"

    if workers == 'auto':
        cli_logger.warning('Getting optimal number of workers based on available memory and inputed networks sizes...')
        workers = __get_optimal_workers(
            nets = nets_path,
            directed= dir,
            comments= comments,
            delimiter= delimiter,
            nets_file_format= nets_file_format
        )
    usable_threads = cpu_count() - 1
    cli_logger.warning(f'Multiprocessing enabled in {workers} out of {usable_threads} usable threads detected')
    
    # Networks topological characterization
    if compare_to_models:
        scalars_array, dist_array, avg_nets_scalars_arrays, avg_nets_moments_arrays = compare_structure(
            networks= nets_path, 
            norm= norm, 
            workers= workers, 
            selected_props= selected_props,
            association_metric= association_metric,
            method= method,
            compare_to_models= compare_to_models,
            n_random_models= n_random_models,
            ba_m= ba_m,
            verbose= verbose,
            return_prop_dicts= True,
            directed= dir,
            nets_file_format= nets_file_format,
            delimiter= delimiter,
            comments= comments,
            keep_averages= keep_averages,
            title= title
        )
    else:
        scalars_array, dist_array = compare_structure(
            networks= nets_path, 
            norm= norm, 
            workers= workers, 
            selected_props= selected_props,
            association_metric= association_metric,
            method= method,
            verbose= verbose,
            return_prop_dicts= True,
            directed= dir,
            nets_file_format= nets_file_format,
            delimiter= delimiter,
            comments= comments,
            keep_averages= keep_averages,
            title= title
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
        if compare_to_models:
            for net_id, props in avg_nets_scalars_arrays.items():
                save_prop_dicts(
                    array= props,
                    net_id= net_id,
                    type= 'scalars',
                    output_dir= output,
                    delimiter= delimiter,
                    cl= cl
                )
            for net_id, props in avg_nets_moments_arrays.items():
                save_prop_dicts(
                    array= props,
                    net_id= net_id,
                    type= 'moments',
                    output_dir= output,
                    delimiter= delimiter,
                    cl= cl
                )
    else: # Plotting is required
        scalars_array = common_props_dict(scalars_array)
        
        if len(scalars_array) > 0 and len(list(scalars_array.values())[0]) > 1:
            if compare_to_models:
                # Getting abbreviations for filtered names
                abbreviations = get_models_abbreviations(avg_nets_scalars_arrays)
                avg_nets_scalars_arrays = common_props_dict(avg_nets_scalars_arrays)
                scalars_array.update(avg_nets_scalars_arrays)
                scalars_array = common_props_dict(scalars_array)
            
            association_df = association(scalars_array, corr_func= association_metric)
            
            if compare_to_models:
                association_df = filter_association_df_for_models(association_df, abbreviations)
            
            association_df = clean_names_association_df(association_df)
            fig_scalars = create_comp_heatmap(association_df, metric= metric, method= method, title= title, verbose= verbose, compare_to_models= True if compare_to_models else False)
            save_figs(fig_scalars, output_dir= output)
            association_df.to_csv(concat_path(output, f'association_df.{exts[delimiter]}'), sep= delimiter)
        else:
            cli_logger.critical("Not enough data to compare. Probably input directory has only one network file.")
            raise ValueError("Not enough data to compare.")

def runmode3(args):
    gs = args.gold_standard
    inferences = args.inferences
    directed = args.directed
    greater_is_better = args.greater_is_better
    keep_auc_coords_dicts = args.keep_auc_coords_dicts
    cutoff = args.cutoff
    score = args.score
    baseline = args.baseline
    self_loops = args.self_loops
    verbose = args.verbose
    output = args.output
    comments = args.comments
    delimiter = args.delimiter

    cl = f'{comments}command: netective {RUNMODES[args.runmode]} --gold_standard {gs} --inferences {inferences} --directed {directed} --greater_is_better {greater_is_better} --keep_auc_coords_dicts {keep_auc_coords_dicts} --cutoff {cutoff} --score {score} --self_loops {self_loops} --baseline {baseline} --verbose {verbose} --output {output} --delimiter {repr(delimiter)} --comments {comments}'
    if keep_auc_coords_dicts: # Keeping dictionaries with aupr and auroc values
        aupr_scores, auroc_scores, coords = benchmarking(
            networks= inferences,
            gold_standard= gs,
            directed= directed,
            greater_score_is_better= greater_is_better,
            allow_self_loops= self_loops,
            cutoff= cutoff,
            baseline= baseline,
            return_auc_coords_dicts= True,
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
            pre_output_file = concat_path(output, f'precision.{ext}')
            sen_output_file = concat_path(output, f'sensitivity.{ext}')
            fpr_output_file = concat_path(output, f'fpr.{ext}')

            aupr_f = open(aupr_output_file, 'w')
            auroc_f = open(auroc_output_file, 'w')
            pre_f = open(pre_output_file, 'w')
            sen_f = open(sen_output_file, 'w')
            fpr_f = open(fpr_output_file, 'w')

            aupr_f.write(f'{cl}\n{comments}AUPR Scores\n{comments}Network{delimiter}Score')
            auroc_f.write(f'{cl}\n{comments}AUROC Scores\n{comments}Network{delimiter}Score')
            pre_f.write(f'{cl}\n{comments}Precision datapoints for every inference\n')
            sen_f.write(f'{cl}\n{comments}Sensistivity datapoints for every inference\n')
            fpr_f.write(f'{cl}\n{comments}False positive rate datapoints for every inference\n')
            
            for _, value in coords.items():
                for i, dist in enumerate(value):
                    if i == 0:
                        pre_f.write(np.array2string(dist, separator= delimiter).replace('\n', '').replace('[','').replace(']',''))
                        pre_f.write('\n')
                    elif i == 1:
                        sen_f.write(np.array2string(dist, separator= delimiter).replace('\n', '').replace('[','').replace(']',''))
                        sen_f.write('\n')
                    else:
                        fpr_f.write(np.array2string(dist, separator= delimiter).replace('\n', '').replace('[','').replace(']',''))
                        fpr_f.write('\n')

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
            sen_f.close()
            pre_f.close()
            fpr_f.close()

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
            baseline= baseline,
            return_auc_coords_dicts= False,
            comments= comments,
            delimiter= delimiter,
            score= score,
            verbose= verbose
        )

        if output is None:
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
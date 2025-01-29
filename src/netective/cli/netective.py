import os
import numpy as np
from multiprocessing import cpu_count

from netective.cli._arrguments import _parse_arguments
from netective.utils import save_prop_dicts, save_figs, common_props_dict, association, parse_network, clean_names_association_df, get_models_abbreviations, filter_association_df_for_models
from netective.structure.dataviz import create_comp_heatmap, plot_distributions, plot_scalars
from netective import compare_structure, characterize_network, benchmarking, classify_networks
from netective.structure.structure import __get_optimal_workers

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

    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --dir {dir} --workers {workers} --selected_props {selected_props} --verbose {verbose} --nets_f_format {nets_file_format} --comments {comments} --delimiter {delimiter!r} --output {output}\n"

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
    include_models = args.incmodels
    compare_to_models = args.comp2models
    n_random_models = args.n_models
    directed_models = args.directed_models
    ba_m = args.m4ba
    keep_averages = args.no_dist_averages

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

    if include_models:
        cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --directed {dir} --selected_props {selected_props} --association {association_metric} --metric {metric} --method {method} --incmodels {include_models} --comp2models {compare_to_models} --n_models {n_random_models} --directed_models {directed_models} --m4ba {ba_m} --workers {workers} --verbose {verbose} --net_f_format {nets_file_format} --comments {comments} --delimiter {delimiter!r} --title {title} --output {output}\n"
    else:
        cl = f"{comments}command: netective {RUNMODES[args.runmode]} --path {nets_path} --norm {norm} --directed {dir} --selected_props {selected_props} --association {association_metric} --metric {metric} --method {method} --workers {workers} --verbose {verbose} --net_f_format {nets_file_format} --comments {comments} --delimiter {delimiter!r} --title {title} --output {output}\n"

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
    if include_models:
        scalars_array, dist_array, avg_nets_scalars_arrays, avg_nets_moments_arrays = compare_structure(
            networks= nets_path, 
            norm= norm, 
            workers= workers, 
            selected_props= selected_props,
            association_metric= association_metric,
            method= method,
            include_models= include_models,
            compare_to_models= compare_to_models,
            n_random_models= n_random_models,
            directed_models= directed_models,
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
        if include_models:
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
            if include_models:
                # In case there are random models that generate directed networks
                avg_nets_scalars_arrays = common_props_dict(avg_nets_scalars_arrays)
                
                scalars_array.update(avg_nets_scalars_arrays)
                scalars_array = common_props_dict(scalars_array)
            
            association_df = association(scalars_array, corr_func= association_metric)
            
            if compare_to_models:
                # Getting abbreviations for filtered names
                abbreviations = get_models_abbreviations(avg_nets_scalars_arrays)
                association_df = filter_association_df_for_models(association_df, abbreviations)
            
            association_df = clean_names_association_df(association_df)
            fig_scalars = create_comp_heatmap(association_df, metric= metric, method= method, title= title, verbose= verbose, compare_to_models= compare_to_models)
            save_figs(fig_scalars, output_dir= output)
            association_df.to_csv(concat_path(output, f'association_df.{exts[delimiter]}'), sep= delimiter)
        else:
            cli_logger.critical("Not enough data to compare. Probably input directory has only one network file.")
            raise ValueError("Not enough data to compare.")

def runmode3(args):
    # Required args
    gs = args.gold_standard
    inferences = args.inferences
    # Optional args
    directed = args.directed
    greater_is_better = args.greater_is_better
    keep_auc_coords_dicts = args.keep_auc_coords_dicts
    cutoff = args.cutoff
    optimal_cutoff = args.optimal_cutoff
    f1_score = args.f1_score
    mcc = args.matthews_corr_coeff
    score = args.score
    baseline = args.baseline
    self_loops = args.self_loops
    verbose = args.verbose
    output = args.output
    comments = args.comments
    delimiter = args.delimiter
    cl = f'{comments}command: netective {RUNMODES[args.runmode]} --gold_standard {gs} --inferences {inferences} --directed {directed} --greater_is_better {greater_is_better} --keep_auc_coords_dicts {keep_auc_coords_dicts} --cutoff {cutoff} --optimal_cutoff {optimal_cutoff} --f1_score {f1_score} --matthews_corr_coeff {mcc} --score {score} --self_loops {self_loops} --baseline {baseline} --verbose {verbose} --output {output} --delimiter {delimiter!r} --comments {comments}\n'

    # Benchmarking call
    output_results = benchmarking(
        networks= inferences,
        gold_standard= gs,
        directed= directed,
        greater_score_is_better= greater_is_better,
        allow_self_loops= self_loops,
        cutoff= cutoff,
        optimal_cutoff= optimal_cutoff,
        f1_score= f1_score,
        mcc= mcc,
        baseline= baseline,
        return_auc_coords_dicts= keep_auc_coords_dicts,
        comments= comments,
        delimiter= delimiter,
        score= score,
        verbose= verbose
    )

    # Managing outputs
    if keep_auc_coords_dicts: # Desired ouput: text files
        aupr_scores = output_results[0]
        auroc_scores = output_results[1]
        coords = output_results[2]
        f1_scores_dist = output_results[3]
        mcc_scores_dist = output_results[4]
        accuracy_dist = output_results[5]
        summary = output_results[6]
        
        # Validating output dir and file extensions
        if not os.path.isdir(output):
            cli_logger.warning(f'Invalid output {output}, setting current directory instead')
            output = os.getcwd()
        exts = {",": "csv", "\t": "tsv"}
        ext = exts.get(args.delimiter, "txt")

        # AUC files (PR and ROC)
        aupr_output_file = concat_path(output, f"aupr_scores.{ext}")
        auroc_output_file = concat_path(output, f"auroc_scores.{ext}")
        aupr_f = open(aupr_output_file, 'w')
        auroc_f = open(auroc_output_file, 'w')
        aupr_f.write(f'{cl}{comments}AUPR Scores\nNetwork{delimiter}Score')
        auroc_f.write(f'{cl}{comments}AUROC Scores\nNetwork{delimiter}Score')
        for net, value in aupr_scores.items():
            aupr_f.write(f'\n{net}{delimiter}{value}')
        for net, value in auroc_scores.items():
            auroc_f.write(f'\n{net}{delimiter}{value}')
        aupr_f.close()
        auroc_f.close()

        # Datapoints files (precision, sensitivity and fpr)
        pre_output_file = concat_path(output, f'precision.{ext}')
        sen_output_file = concat_path(output, f'sensitivity.{ext}')
        fpr_output_file = concat_path(output, f'fpr.{ext}')
        pre_f = open(pre_output_file, 'w')
        sen_f = open(sen_output_file, 'w')
        fpr_f = open(fpr_output_file, 'w')
        pre_f.write(f'{cl}{comments}Precision datapoints for every inference\n')
        sen_f.write(f'{cl}{comments}Sensitivity datapoints for every inference\n')
        fpr_f.write(f'{cl}{comments}False positive rate datapoints for every inference\n')
        for net_name, value in coords.items():
            pre_f.write(f'>>>{net_name} precision datapoints\n')
            sen_f.write(f'>>>{net_name} sensitivity datapoints\n')
            fpr_f.write(f'>>>{net_name} false positive rate datapoints\n')
            for i, coords_array in enumerate(value):
                for value in coords_array:
                    if i == 0:
                        pre_f.write(f'{value}{delimiter}')
                    elif i == 1:
                        sen_f.write(f'{value}{delimiter}')
                    else:
                        fpr_f.write(f'{value}{delimiter}')
            pre_f.write('\n')
            sen_f.write('\n')
            fpr_f.write('\n')
        sen_f.close()
        pre_f.close()
        fpr_f.close()

        # Stats summary file 
        stats_summary_file = concat_path(output, f'stats_summary.{ext}')
        stats_summary_f = open(stats_summary_file, 'w')
        stats_summary_f.write(f'{cl}{comments}Stats summary for every inference\n')
        stats_summary_f.close()
        summary.to_csv(
            path_or_buf= stats_summary_file,
            sep= delimiter,
            na_rep= 'NaN',
            mode= 'a'
        )

        # Scores-metrics distributions
        scores_metrics_file = concat_path(output, f'scores_metrics_distributions.{ext}')
        scores_metrics_f = open(scores_metrics_file, 'w')
        scores_metrics_f.write(f'{cl}{comments}Scores-metric distribution for every inference for F1 score, MCC and Accuracy\n')
        net_names = list(f1_scores_dist.keys())
        for net_name in net_names:
            if net_name == 'Baseline':
                continue
            scores_metrics_f.write(f'>>>{net_name} inference scores\n')
            # Write scores
            for score in f1_scores_dist[net_name].keys():
                scores_metrics_f.write(f'{score}{delimiter}')
            # F1 scores
            scores_metrics_f.write(f'\n>>>{net_name} F1 scores\n')
            for score in f1_scores_dist[net_name].values():
                scores_metrics_f.write(f'{score}{delimiter}')
            # MCC values
            scores_metrics_f.write(f'\n>>>{net_name} Matthews Correlation Coefficient values\n')
            for score in mcc_scores_dist[net_name].values():
                scores_metrics_f.write(f'{score}{delimiter}')
            # Accuracy scores
            scores_metrics_f.write(f'\n>>>{net_name} Accuracy scores\n')
            for score in accuracy_dist[net_name].values():
                scores_metrics_f.write(f'{score}{delimiter}')
            scores_metrics_f.write('\n')
        scores_metrics_f.close()

    else: # Desired ouput: plotting
        output_results['aupr'].get_figure().savefig(fname= concat_path(output, f'aupr.png'), bbox_inches= 'tight', dpi= 300)
        output_results['pr curves'].get_figure().savefig(fname= concat_path(output, f'pr_curves.png'), bbox_inches= 'tight', dpi= 300)
        output_results['auroc'].get_figure().savefig(fname= concat_path(output, f'auroc.png'), bbox_inches= 'tight', dpi= 300)
        output_results['roc curves'].get_figure().savefig(fname= concat_path(output, f'roc_curves.png'), bbox_inches= 'tight', dpi= 300)
        if optimal_cutoff:
            output_results['optimal cutoffs'].get_figure().savefig(fname= concat_path(output, f'optimal_cutoffs.png'), bbox_inches= 'tight', dpi= 300)
        if f1_score:
            output_results['f1 scores'].get_figure().savefig(fname= concat_path(output, f'f1_scores.png'), bbox_inches= 'tight', dpi= 300)
        if mcc:
            output_results['mcc'].get_figure().savefig(fname= concat_path(output, f'mcc.png'), bbox_inches= 'tight', dpi= 300)

def runmode4(args):
    # Required args
    distance_df_path = args.distance_dataframe
    results_dir = args.results_dir
    networks_dir = args.networks_dir
    # Optional args
    norm = args.normalization
    directed = args.directed
    selected_props = args.selected_props
    workers = args.workers
    add_averages = args.add_averages
    association_metric = args.association_metric
    clust_num = args.clusters
    threshold = args.threshold if clust_num is None else None
    metric = args.metric
    method = args.method
    map_ids = args.map_ids
    verbose = args.verbose
    net_file_format = args.net_f_format
    comments = args.comments
    delimiter = args.delimiter
    output = args.output
    cl = f"{comments}command: netective {RUNMODES[args.runmode]} --distance_dataframe {distance_df_path} --results_dir {results_dir} --networks_dir {networks_dir} --normalization {norm} --directed {directed} --selected_props {selected_props} --workers {workers} --add_averages {add_averages} --association_metric {association_metric} --clusters {clust_num} --threshold {threshold} --metric {metric} --method {method} --map_ids {map_ids} --verbose {verbose} --net_f_format {net_file_format} --comments {comments} --delimiter {delimiter!r} --output {output}\n"
    
    if distance_df_path:
        distance_df = pd.read_csv(distance_df_path, comment= comments, delimiter= delimiter, index_col= 0, header= 0)
    else:
        distance_df = None
    clusters = classify_networks(
        networks= networks_dir,
        results_dir= results_dir,
        distance_df= distance_df,
        norm= norm,
        directed= directed,
        selected_props= selected_props,
        workers= workers,
        add_averages= add_averages,
        association_metric= association_metric,
        clust_num= clust_num,
        threshold= threshold,
        metric= metric,
        method= method,
        map_ids= True,
        verbose= verbose,
        nets_file_format= net_file_format,
        comments= comments,
        delimiter= delimiter
    )
    # File creation
    if not os.path.isdir(output):
        cli_logger.warning(f'Invalid output {output}, setting current directory instead')
        output = os.getcwd()
    exts = {",": "csv", "\t": "tsv"}
    ext = exts.get(args.delimiter, "txt")
    output_file = concat_path(output, f"networks_classification.{ext}")
    cli_logger.info(f'Writing output file with results: {output_file}')
    f = open(output_file, 'w')
    f.write(f'{cl}Network{args.delimiter}Cluster')
    
    for clust, nets in clusters.items():
        for net in nets:
            f.write(f'\n{net}{args.delimiter}{clust}')
    f.close()

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
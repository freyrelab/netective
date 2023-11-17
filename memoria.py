import os
from netective import characterize_network, compare_structure
import tracemalloc
import pandas as pd
from netective.utils import concat_path, parse_network, save_figs, sort_files, common_props_dict, association
from netective.structure.dataviz import plot_distributions, plot_scalars, create_symmetric_heatmap


def get_allocated_memory(snapshot, key_type='lineno', filtered: bool = True):
    if filtered:
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(True, '*networkx*')
        ))
    top_stats = snapshot.statistics(key_type)
    return ((sum(stat.size for stat in top_stats)) / 1024) * 0.001024

def main():
    runmode = 2

    if runmode == 1:

        overall_info = {
            'File Size' : [],
            'Net' : [],
            'Nodes' : [],
            'Edges' : [],
            'Memory Networkx (Mbs)' : [],
            'Overall memory (Mbs)' : []
        }


        files = sort_files(path='./data/only_abasy/')

        for i,f in enumerate(files):
            if i < len(files):
                print(f'Analyzing {i+1}/{len(files)}')
                tracemalloc.start()
                net_id = os.path.basename(f)
                net = parse_network(file_path= f, delimiter=' ', directed=True)
                # Llamada a characterize
                foo, spam = characterize_network(
                    G= net,
                    net_id= net_id,
                    verbose='critical',
                    return_prop_dicts= True
                )
                for net_id, props in foo.items():
                    fig_scalar, _ = plot_scalars(data_dict= props, verbose= 'critical')
                    save_figs(
                        fig= fig_scalar,
                        props= 'scalars',
                        net_id= net_id,
                        compare= False,
                        output_dir= './results_dummy/plots/dummy_plots/'
                    )
                for net_id, props in spam.items():
                    fig_dist, _ = plot_distributions(props, verbose= 'critical')
                    save_figs(
                        fig= fig_dist,
                        props= 'distributions',
                        net_id= net_id,
                        compare= False,
                        output_dir= './results_dummy/plots/dummy_plots/'
                    )
                print(f'\tPlotted {i+1}/{len(files)}')
                snapshot = tracemalloc.take_snapshot()
                overall_info['File Size'].append(os.stat(f).st_size)
                overall_info['Net'].append(net_id)
                overall_info['Nodes'].append(net.number_of_nodes())
                overall_info['Edges'].append(net.number_of_edges())
                overall_info['Memory Networkx (Mbs)'].append(get_allocated_memory(snapshot))
                overall_info['Overall memory (Mbs)'].append(get_allocated_memory(snapshot, filtered= False))

        df = pd.DataFrame.from_dict(overall_info)
        df.to_csv('./results_dummy/optimized_allocated_memory_abasy.csv', header= True, index= False)

    elif runmode == 2:

        overall_info = {
        'Workers' : [],
        'Memory Networkx (Mbs)' : [],
        'Overall memory (Mbs)' : []
        }

        for workers in range(2, 6):
            
            tracemalloc.start()
            nets_to_submit = f'./data/test_{workers}'
            scalars_array, _ = compare_structure(
                networks= nets_to_submit,
                norm= 'network',
                return_prop_dicts= True,
                workers= workers,
                keep_averages= True,
                delimiter= ' '
            )

            scalars_array = common_props_dict(scalars_array)
            if len(scalars_array) > 0 and len(list(scalars_array.values())[0]) > 1:
                df = association(scalars_array)
                fig_scalars = create_symmetric_heatmap(df, title=f"Global properties")
                save_figs(fig_scalars, output_dir= './results_dummy/plots/dummy_plots/')
            
            snapshot = tracemalloc.take_snapshot()
            overall_info['Workers'].append(workers)
            overall_info['Memory Networkx (Mbs)'].append(get_allocated_memory(snapshot))
            overall_info['Overall memory (Mbs)'].append(get_allocated_memory(snapshot, filtered= False))
        
        df = pd.DataFrame.from_dict(overall_info)
        df.to_csv('./results_dummy/compare_run_5nets.csv', header= True, index= False)

if __name__ == '__main__':
    main()








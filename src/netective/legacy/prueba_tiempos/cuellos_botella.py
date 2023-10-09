import pickle
from tqdm import tqdm
import concurrent.futures
import os
import networkx as nx
import numpy as np
import re
from netective.structure.struct_dummy import Structure
import warnings
warnings.filterwarnings("ignore")
import time


def run_parallel(f, my_iter, workers):
    len_iter = len(my_iter)
    with tqdm(total=len_iter) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for arg in my_iter:
                path = arg[0].split('\\')
                if arg[1] == 'INF_':
                    name = arg[1] + path[len(path) - 2] + '_' + path[len(path) - 1].split('.')[0]
                else:
                    name = arg[1] + path[len(path) - 1].split('.')[0]
                if path[len(path) - 1].split('.')[1] != 'txt':
                    name = name + '.' + path[len(path) - 1].split('.')[1]
                print(f'\n\nSubmitting net: {name}')
                futures[executor.submit(f, *(arg[0], name))] = name

            results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                    print(f'\nFinilized: {futures[future]}')
                    pbar.update(1)
                except Exception as exc:
                    print(f"Error: {exc}")
    return results

def get_paths(directory_name= str, validate= False, selected= list):
    paths = []
    for root, dir, files in os.walk(directory_name):
        if len(files) != 0:
            for f in files:
                if validate:
                    if re.search('.txt', f) and f in selected:
                        path = os.path.join(os.getcwd(), root, f)
                        paths.append(path)
                else:
                    if re.search('.txt', f) or re.search('.genes', f):
                        path = os.path.join(os.getcwd(), root, f)
                        paths.append(path)

    return paths

def analyze(net_path= str, name= str):
    scalar_raw = {}
    scalar_networknorm = {}

    with open(net_path) as f:
        net = nx.DiGraph([line.split()[:2] for line in f if not re.search('regulator', line)])

    struct = Structure(net, norm=None, net_id=name, verbose=False)
    inicio_get_props = time.time()
    scalar_values, dist_props, prop_times = struct.get_props()
    fin_get_props = time.time()
    prop_times['get_props'] = fin_get_props - inicio_get_props
    prop_times['edges'] = net.number_of_edges()
    prop_times['nodes'] = net.number_of_nodes()

    scalar_raw[name] = scalar_values[name]
    for prop_name, prop_moments in dist_props[name].items():
        scalar_raw[name][f'Average {prop_name}'] = prop_moments[0]
    
    struct.norm = 'network'
    inicio_get_props = time.time()
    scalar_values, dist_props, norm_times = struct.get_props()
    fin_get_props = time.time()
    norm_times['get_props'] = fin_get_props - inicio_get_props
    norm_times['edges'] = net.number_of_edges()
    norm_times['nodes'] = net.number_of_nodes()
    
    scalar_networknorm[name] = scalar_values[name]
    for prop_name, prop_moments in dist_props[name].items():
        scalar_networknorm[name][f'Average {prop_name}'] = prop_moments[0]
    
    data_dict = {
        f'raw_props_{name}' : prop_times,
        f'norm_props_{name}' : norm_times
    }

    return data_dict

if __name__ == '__main__':
    """
    final_nets = [
            'yeast_chipunion_KDUnion_intersect.txt',
            'yeast_chipunion.txt',
            'yeast_KDUnion.txt',
            'mESC_KDUnion.txt',
            'mESC_chipunion.txt',
            'mESC_chipunion_KDUnion_intersect.txt',
            'mDC_KDUnion.txt',
            'mDC_chipunion.txt',
            'mDC_chipunion_KDUnion_intersect.txt',
            'hESC_KDUnion.txt',
            'hESC_chipunion.txt',
            'hESC_chipunion_KDUnion_intersect.txt'
        ]
    """
    test_nets = [
         'yeast_chipunion_KDUnion_intersect.txt',
         'mESC_chipunion_KDUnion_intersect.txt',
         'mDC_chipunion_KDUnion_intersect.txt',
         'hESC_chipunion_KDUnion_intersect.txt'
    ]
    paths = get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\netective\\data\\mouse_net")
    # paths.extend(get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\netective\\data\\human_net"))
    # paths = get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\aux_netective\\single_cell_analysis\\gold_standard_datasets", validate= True, selected= test_nets)
    # paths.extend(get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\aux_netective\\single_cell_analysis\\imputed_inferred_networks"))

    test = [path for i,path in enumerate(paths) if i < 2]

    # input_dataset = [(net_path, 'GS_') if i < 12 else (net_path, 'INF_') for i,net_path in enumerate(paths)]
    input_dataset = [(net_path, 'INF_') for net_path in test]
    raw_props_times = {}
    norm_props_times = {}

    for j,(name, data_dict) in enumerate(run_parallel(analyze, input_dataset, workers=2).items()):
        for i,(name_2, data) in enumerate(data_dict.items()):
            dict_ = raw_props_times if re.search('raw', name_2) else norm_props_times
            if i == 0 and j == 0:
                for prop in data.keys():
                    dict_[prop] = []
            elif i == 1 and j == 0:
                for prop in data.keys():
                    dict_[prop] = []
            for prop, value in data.items():
                dict_[prop].append(value)
    
    
    with open(f'results\\raw_times.pkl', 'wb') as f:
                pickle.dump(raw_props_times, f)
    
    with open(f'results\\norm_times.pkl', 'wb') as f:
                pickle.dump(norm_props_times, f)     
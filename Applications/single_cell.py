import pickle
import inspect
from math import log
from tqdm import tqdm
import concurrent.futures
import os
import pandas as pd
import networkx as nx
import numpy as np
import re
from netective.structure.structure import Structure
import warnings
warnings.filterwarnings("ignore")


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
                print(f'Submitting net: {name}')
                futures[executor.submit(f, *(arg[0], name))] = name

            results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                    print(f'Finilized: {futures[future]}')
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
                    if re.search('.txt', f):
                        path = os.path.join(os.getcwd(), root, f)
                        paths.append(path)

    return paths

def analyze(net_path= str, name= str):
    scalar_raw = {}
    scalar_networknorm = {}

    with open(net_path) as f:
        net = nx.DiGraph([line.split()[:2] for line in f if not re.search('regulator', line)])
        print(f'Net: {name}, Nodes: {net.number_of_nodes()}, Edges: {net.number_of_edges()}')
    struct = Structure(net, norm=None, net_id=name, verbose=False)
    scalar_values, dist_props = struct.get_props()
    scalar_raw[name] = scalar_values[name]
    for prop_name, prop_moments in dist_props.items():
        scalar_raw[name][f'Average {prop_name}'] = prop_moments[0]
    
    struct.norm = 'network'
    scalar_values, dist_props = struct.get_props()
    scalar_networknorm[name] = scalar_values[name]
    for prop_name, prop_moments in dist_props.items():
        scalar_networknorm[name][f'Average {prop_name}'] = prop_moments[0]
    
    data_dict = {
        f'scalar_raw_{name}' : scalar_raw,
        f'scalar_nwtworknorm_{name}' : scalar_networknorm
    }

    return data_dict

if __name__ == '__main__':
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
    paths = get_paths('./gold_standard_datasets', validate= True, selected= final_nets)
    paths.extend(get_paths('./imputed_inferred_networks/'))

    test = [path for i,path in enumerate(paths) if i < 20]

    input_dataset = [(net_path, 'GS_') if i < 12 else (net_path, 'INF_') for i,net_path in enumerate(test)]

    run_parallel(analyze, input_dataset, workers = 5)
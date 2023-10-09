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
import random
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
                    if re.search('.txt', f) or re.search('.genes', f):
                        path = os.path.join(os.getcwd(), root, f)
                        paths.append(path)

    return paths

def analyze(net_path= str, name= str):
    inicio = time.time()
    scalar_raw = {}
    scalar_networknorm = {}
    er_scalar_raw = {}
    er_scalar_networknorm = {}

    
    print(f'\t\tEmpieza a leer el archivo {time.time() - inicio}')
    with open(net_path) as f:
        net = nx.DiGraph([line.split()[:2] for line in f if not re.search('regulator', line)])
    print(f'\t\tTermina de cargar la red {time.time() - inicio}')

    seed = 42
    random.seed(seed)
    density = net.number_of_edges() / (net.number_of_nodes() * (net.number_of_nodes() -1))
    er_net = nx.erdos_renyi_graph(n= net.number_of_nodes(), p= density, directed= True, seed=seed)
    print(f'\t\tSe generó la ER del mismo tamaño {time.time() - inicio}')

    print(f'\t\tOG net: {name}, nodes: {net.number_of_nodes()}, edges: {net.number_of_edges()}')
    
    print(f'\t\tER net, nodes: {er_net.number_of_nodes()}, edges: {er_net.number_of_edges()}')

    print(f'\t\tSe inicia la instancia de la red original {time.time() - inicio}')
    struct = Structure(net, norm=None, net_id=name, verbose=False)
    print(f'\t\tSe terminó de instanciar {time.time() - inicio}')

    print(f'\tEntra a get props para {name}, norm: raw {time.time() - inicio}')
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
    print(f'\tEntra a get props para {name}, norm: network {time.time() - inicio}')
    scalar_values, dist_props = struct.get_props()
    scalar_networknorm[name] = scalar_values[name]
    for prop_name, prop_moments in dist_props[name].items():
        scalar_networknorm[name][f'Average {prop_name}'] = prop_moments[0]

    print(f'\t\tSe inicia la instancia con la er{time.time() - inicio}')
    struct = Structure(er_net, norm=None, net_id=f'ER_{name}', verbose=False)
    print(f'\t\tTermina la instancia de la er {time.time() - inicio}')
    print(f'\tEntra a get props para ER, norm: raw {time.time() - inicio}')
    scalar_values, dist_props = struct.get_props()
    er_scalar_raw[f'ER_{name}'] = scalar_values[f'ER_{name}']
    for prop_name, prop_moments in dist_props[f'ER_{name}'].items():
        er_scalar_raw[f'ER_{name}'][f'Average {prop_name}'] = prop_moments[0]
    
    struct.norm = 'network'
    print(f'\tEntra a get props para ER, norm: network {time.time() - inicio}')
    scalar_values, dist_props = struct.get_props()
    er_scalar_networknorm[f'ER_{name}'] = scalar_values[f'ER_{name}']
    for prop_name, prop_moments in dist_props[f'ER_{name}'].items():
        er_scalar_networknorm[f'ER_{name}'][f'Average {prop_name}'] = prop_moments[0]
    
    print(f'\tSe genera el dict final {time.time() - inicio}')
    data_dict = {
        f'scalar_raw_{name}' : scalar_raw,
        f'scalar_networknorm_{name}' : scalar_networknorm,
        f'er_scalar_raw_{name}' : er_scalar_raw,
        f'er_scalar_networknorm_{name}' : er_scalar_networknorm
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
    # paths = get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\netective\\data\\mouse_net")
    paths = get_paths("Data\\gold_standard_datasets", validate= True, selected= final_nets)
    # paths.extend(get_paths("H:\\Mi unidad\\Respaldo\\Genomicas\\aux_netective\\single_cell_analysis\\imputed_inferred_networks"))

    # test = [path for i,path in enumerate(paths) if i < 1]
    test = [x for x in paths if x.endswith('mDC_chipunion_KDUnion_intersect.txt')]

    input_dataset = [(net_path, 'GS_') if i < 12 else (net_path, 'INF_') for i,net_path in enumerate(test)]
    # input_dataset = [(net_path, 'INF_') for net_path in test]

    for name, data_dict in run_parallel(analyze, input_dataset, workers=5).items():
        for name_2, data in data_dict.items():
            with open(f'results\\{name_2}.pkl', 'wb') as f:
                pickle.dump(data, f)
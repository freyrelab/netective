import pickle
import inspect
import numpy as np
from math import log
from tqdm import tqdm
import networkx as nx
import concurrent.futures

from netective.structure.structure import Structure, characterize_network
from netective.structure import properties
parent_class = properties._Property

import warnings
warnings.filterwarnings("ignore")

def run_parallel(f, my_iter, workers):
    len_iter = len(my_iter)
    with tqdm(total=len_iter) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for arg in my_iter:
                # print(f'Running: {arg[0]} nodes, {arg[1]} density, {arg[2]} random times each model...')
                name = f'{arg[0]}_{arg[1]}'
                futures[executor.submit(f, *arg)] = name

            results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                    pbar.update(1)
                except Exception as exc:
                    print(f"Error: {exc}")
    return results



def synthetic_charact(num_nodes, density, random_times):

    child_classes = {}
    for name, obj in inspect.getmembers(properties):
        if inspect.isclass(obj) and issubclass(obj, parent_class) and obj != parent_class:
            if obj._use_direction:
                continue
            bool_mask = [
                    obj._use_direction,
                    obj._use_selfloops,
                    obj._use_giant_component,
                    obj._use_paths,
                ]
            child_classes[obj] = np.packbits(bool_mask).item() >> 4

    nodes = {}
    edges = {}
    scalar_raw = {}
    scalar_networknorm = {}
    k = int(round(density*num_nodes / 2))    # Watts-Strogatz k, Barabasi-Albert m, number of edges to attach from a new node to existing nodes
    k = 2 if k < 2 else k # k must be at least 2 for WS
    dgm_gen_n = int(round(log(2/3, 3) + log(num_nodes,3) + 1))

    for i in range(random_times):
        networks = {}
        networks['ba_graph'] = nx.barabasi_albert_graph(n=num_nodes, m=k, seed=i)
        networks['er_graph'] = nx.erdos_renyi_graph(n=num_nodes, p=density, seed=i)
        networks['ws_graph'] = nx.watts_strogatz_graph(n=num_nodes, k=k, p=0.5, seed=i)
        networks['dgm_graph'] = nx.dorogovtsev_goltsev_mendes_graph(n=dgm_gen_n) # already deterministic
        networks['sf_graph'] = nx.Graph(nx.scale_free_graph(n=num_nodes, seed=i)) # original scale-free graph is MultiGraph


        for name, network in networks.items():

            network = nx.DiGraph(network)

            nodes[f'{name}_{density}_{num_nodes}_{i}'] = network.number_of_nodes()
            edges[f'{name}_{density}_{num_nodes}_{i}'] = network.number_of_edges()

            struct = Structure(network, norm=None, net_id=name, verbose=False)
            scalar_values, dist_props = struct.get_props(child_classes=child_classes)
            scalar_raw[f'{name}_{density}_{num_nodes}_{i}'] = scalar_values[name]
            for prop_name, prop_values in dist_props[name].items():
                scalar_raw[f'{name}_{density}_{num_nodes}_{i}'][f'Average {prop_name}'] = prop_values[0]
                scalar_raw[f'{name}_{density}_{num_nodes}_{i}'][f'Variation {prop_name}'] = prop_values[1]
                scalar_raw[f'{name}_{density}_{num_nodes}_{i}'][f'Skewness {prop_name}'] = prop_values[2]
                scalar_raw[f'{name}_{density}_{num_nodes}_{i}'][f'Kurtosis {prop_name}'] = prop_values[3]

            struct.norm = 'network'
            scalar_values, dist_props = struct.get_props(child_classes=child_classes)
            scalar_networknorm[f'{name}_{density}_{num_nodes}_{i}'] = scalar_values[name]
            for prop_name, prop_values in dist_props[name].items():
                scalar_networknorm[f'{name}_{density}_{num_nodes}_{i}'][f'Average {prop_name}'] = prop_values[0]
                scalar_networknorm[f'{name}_{density}_{num_nodes}_{i}'][f'Variation {prop_name}'] = prop_values[1]
                scalar_networknorm[f'{name}_{density}_{num_nodes}_{i}'][f'Skewness {prop_name}'] = prop_values[2]
                scalar_networknorm[f'{name}_{density}_{num_nodes}_{i}'][f'Kurtosis {prop_name}'] = prop_values[3]
    
    data_dict = {
        f'scalar_networknorm_{density}_{num_nodes}' : scalar_networknorm,
        f'scalar_raw_{density}_{num_nodes}' : scalar_raw,
        f'nodes_{density}_{num_nodes}' : nodes,
        f'edges_{density}_{num_nodes}' : edges
    }
                 
    return data_dict


if __name__ == '__main__':

    random_times = 100  # number of times to run random graph generation
    num_nodes_values = range(1000, 5001, 1000)
    density_values = [0.001, 0.005, 0.01]#, 0.05]

    input_datasets = [(num_nodes, density, random_times) for num_nodes in num_nodes_values for density in density_values]


    for name, data_dict in run_parallel(synthetic_charact, input_datasets, workers=8).items():
        for name_2, data in data_dict.items():
            with open(f'{name_2}.pkl', 'wb') as f:
                pickle.dump(data, f)
from __future__ import annotations
import pickle
import random
import inspect
import numpy as np
from math import log
from tqdm import tqdm
import networkx as nx
import concurrent.futures
from multiprocessing import cpu_count


from freyrelab.abasy import Abasy
from freyrelab.nets import models, dissimilarity
from freyrelab.nets.sampling import edge_sampling, node_sampling, snowball_sampling

from netective.structure.structure import Structure, characterize_network
from netective.structure import properties
parent_class = properties._Property

# import warnings
# warnings.filterwarnings("ignore

def run_parallel(f, my_iter, workers):
    len_iter = len(my_iter)
    with tqdm(total=len_iter) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for arg in my_iter:
                # print(f'Running: {arg[0]}, {arg[3]}% step, {arg[2]} random times each step...')
                name = arg[0]
                futures[executor.submit(f, *arg)] = name

            results = {}
            for future in concurrent.futures.as_completed(futures):
                # try:
                results[futures[future]] = future.result()
                pbar.update(1)
                # except Exception as exc:
                #     print(f"Error: {exc}")
    return results

def get_samples(G: nx.DiGraph,  
                step: int=10, 
                rep: int=10,
                sampling_type : str='node',
                deterministic: bool=False 
):
    """
        Function to create sampling networks.

        Args:
            G: RegNet or DiGraph.
                Network to compute the structural properties.
            step: int.
                Base level of network coverage for sampling and step for each
                increase until reaching 100% of coverage.
                Default is 10.
            rep: int.
                Number of samples per step.
                Default is 10.
            sampling_type: str.
                Network attribute by which sampling is done.
                Sampling types admitted: node-based, edge-based and snowball effect.
                Default is node-based sampling.

        Returns:
            nets : dictionary where keys as IDs for each network and its value is the sampled network.

        Raises:
            TypeError: If G is not a RegNet or DiGraph.
            ValueError: If user introduces a sampling type not admitted.
        """
    

    sampling_opt = ['node', 'edge', 'snowball']
    if sampling_type not in sampling_opt:
        raise ValueError(f'Invalid sampling type {sampling_type}, sampling options: {sampling_opt}')
    
    sampling_fxns = {
        'node' : node_sampling,
        'edge' : edge_sampling,
        'snowball' : snowball_sampling
    }
    

    nets = {}
    for x,i in enumerate(range(step, 101, step), 1):
        for j in range(rep):
            if deterministic:
                random.seed(j)
            size = int(G.number_of_nodes() * i / 100) if sampling_type != 'edge' else int(G.number_of_edges() * i / 100)
            # print(f'Computing {sampling_type} sampling with {i}% coverage and {j} repetition... nodes: {size} and edges: {G.number_of_edges()}')
            nets[f'Net_{sampling_type}_{step * x}_{j}'] = sampling_fxns[sampling_type](G, size=size)
    
    return nets

def sampling_charact(name_id, G, random_times, perc_sampling_step):

    sampled_nets = {}
    for sampling in ['node', 'edge', 'snowball']:
        # print(f'Computing {sampling} sampling... for {name_id}')
        nets = get_samples(G, sampling_type=sampling, deterministic=True, step=perc_sampling_step, rep=random_times)
        nets = {k.replace('Net', name_id) : v for k,v in nets.items()}
        sampled_nets.update(nets)
    
    data_dict = {}
    for i, name in enumerate(sampled_nets):
        data_dict[name] = {}
        for j, name2 in enumerate(list(sampled_nets)[i+1:]):
            try:
                data_dict[name][name2] = dissimilarity.graph_dissimilarity(sampled_nets[name], sampled_nets[name2])
            except:
                # print(f'Error computing dissimilarity between {name} and {name2}')
                data_dict[name][name2] = np.nan

    return data_dict


def main():
    random_times = 10
    perc_sampling_step = 10

    a = Abasy()
    net_ids = [
        '208964_v2020_sRPA20_eStrong',
        '511145_v2022_sRDB22_eStrong',
        '100226_v2019_sA22-DBSCR15_eStrong',
        '196627_v2020_s21_eStrong',
        '224308_v2008_sDBTBS08_eStrong'
        ]
    id_net = {}
    for id_ in net_ids:
        H = a.regnet(id_)
        G = nx.DiGraph()
        G.add_nodes_from(H.nodes)
        G.add_edges_from(H.edges)
        id_net[id_] = G

    input_datasets = [(name, G, random_times, perc_sampling_step) for name, G in id_net.items()]

    workers = 5#int((cpu_count()-2)/len(net_ids))
    print(f'Using {workers} workers')
    for name, data_dict in run_parallel(sampling_charact, input_datasets, workers=workers).items():
        with open(f'{name}_dvalue.pkl', 'wb') as f:
            pickle.dump(data_dict, f)

if __name__ == '__main__':
    main()
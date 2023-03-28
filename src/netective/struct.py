from __future__ import annotations

import os
import math
from networkx import DiGraph
from pandas import DataFrame
from warnings import warn

import netbiol3 as nb

from netective.utils import *


def _max_loops(n:int, r:int, tfs:int ,r_tfs:int) -> int:
    """
    Computes the maximum number of motifs of size r with r_tfs TFs in a network of n nodes with tfs TFs.

    Args:
        n = number of nodes in the network
        r = number of elements in the motif
        tfs = number of TFs in the network
        r_tfs = number of TFs in the motif
    
    Returns:
        int: maximum number of motifs of size r with r_tfs TFs in a network of n nodes with tfs TFs.
    """
    putative = math.factorial(n) / math.factorial(n-r)
    # fraccion de TFs al cuadrado para feed forward (el expoente es el numero de TFs en el motivo)
    putative = putative * ( (tfs/n)**r_tfs )
    
    return putative



def struc_props(G: DiGraph | nb.RegNet, net_id: str, norm: bool) -> dict[str, float, int]:

    """
    Computes the structural properties of a network.

    Args:
        G: DiGraph or RegNet.
            Network to compute the structural properties.
        net_id: str.
            Name of the network.
        norm: bool.
            If True, the properties are normalized (biological criteria).

    Returns:
        dict: Dictionary with the structural properties of the network.
    
    Raises:
        TypeError: If G is not a DiGraph or a RegNet.
        ValueError: If G has no edges.

    TODO: Change the following code to use pandas using the properties names as keywords, 
    the first colum as the raw values, another column with the fields to use as the divisor 
    in the normalization. Save the normalized values in another column. Add columns only if 
    required (if norm). Keep the return structures converting the dataframe to a dict.
    """

    G = validate_network(G)
    
    print(f'Processing {net_id}...', flush=True)
    print(f'{net_id} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.', flush=True)

    G.remove_isolates()
    n_genes = len(G)
    props = {}
    
    props['Density'] = G.density
    props['Regulators'] = G.regulators_count
    props['Self regulations'] = G.selfinteractions_count
    props['Max. out connectivity'] = G.kout_max
    
    #in-place
    G.remove_selfinteractions()
    G.remove_isolates()
    
    #Props without selfloops
    props['3-Feedback loops'] = G.feedbacks3_count
    props['Feedforward circuits'] = G.feedforwards_count
    props['Complex feedforward circuits'] = G.complex_feedforwards_count
    props['Genes in the giant component'] = G.giant_component_size
    

    props['Diameter'] = G.diameter()
    props['Average shortest path length'] = G.average_path_length()
    props['Average clustering coefficient'] = G.average_clustering_coefficient
    
    
    #c(k)
    kc = G.k_clustering()

    try:
        CK = nb.Ck(kc.values())
        props['R^2 C(k)'] = CK.rsquared_adj
    except ValueError:
        warn(f'Clusterings for {net_id}: {set(list(zip(*kc.values()))[1])}, cannot be fitted to a power law.')
        props['R^2 C(k)'] = 0   ## TODO: check if this is correct
    
    #p(k)
    try:
        k, _ = zip(*kc.values())
        PK = nb.Pk(k)
        props['R^2 P(k)'] = PK.rsquared_adj
    except ValueError:
        warn(f'Degrees for {net_id}: {set(list(zip(*kc.values()))[1])}, cannot be fitted to a power law.')
        props['R^2 P(k)'] = 0   ## TODO: check if this is correct
    
    # kappa-value # TODO: generates error bc kappa values lower than 1 or nan
    # kc = G.k_clustering(kdir='out')
    # CK = nb.Ck(kc.values())
    # print(net_id, CK.kappa)
    # props['Kappa'] = round(CK.kappa)


    if norm:

        print(f'norm: {norm}')

        warn('Normalization for clustering coefficient not implemented yet')

        # Normalized
        tfs = props['Regulators']
        max_3tfs_loop = _max_loops(n=n_genes, r=3, tfs=tfs, r_tfs=3)
        max_2tfs_loop = _max_loops(n=n_genes, r=3, tfs=tfs, r_tfs=2)
        largest_putative_path = props['Genes in the giant component']-1

        props_n = {
            'Density': props['Density'] * (n_genes/tfs),        # equivalent to E / (n**2 * (tfs/n)) # TODO: corrects previous mistake: G.density * (G.regulators_count / nGenes) << 1
            'Regulators': tfs / n_genes,                        # fraction of nodes that are regulators
            'Self regulations': props['Self regulations']/tfs,  # fraction of regulators that self-regulate
            'Max. out connectivity': props['Max. out connectivity']/n_genes,        # fraction of nodes that are regulated by the most hub
            '3-Feedback loops': props['3-Feedback loops'] / max_3tfs_loop,          # fraction of possible 3-feedback loops
            'Feedforward circuits': props['Feedforward circuits'] / max_2tfs_loop,  # fraction of possible feedforward circuits
            'Complex feedforward circuits': props['Complex feedforward circuits'] / max_2tfs_loop,  # fraction of possible complex feedforward circuits A->B->C, A->C, B->A
            'Genes in the giant component': props['Genes in the giant component'] / n_genes,        # fraction of nodes in the giant component
            'Diameter': props['Diameter'] / largest_putative_path,                  # denominator is equivalent to (n-2+1) for the nodes in the giant component. n-2 (excluding sorce and target nodes) + 1 (we are coiunting edges). # TODO: corrects previous mistake: props['Diameter'] / (n_genes-2) 
            'Average shortest path length': props['Average shortest path length'] / largest_putative_path,  # denominator is equivalent to (n-2+1) for the nodes in the giant component. # TODO corrects previous mistake: / (n_genes-2) 
            'Average clustering coefficient': props['Average clustering coefficient'],  # TODO: This still needs to be normalized
            'R^2 C(k)': props['R^2 C(k)'],   # already normalized
            'R^2 P(k)': props['R^2 P(k)'],   # already normalized
            'Kappa': props['Kappa']/n_genes,    #  Kapa over the k/kmax space
        }
    
        props = props_n
    
    return net_id, props

def struc_props_call(G: DiGraph | nb.RegNet, net_id: str, norm: bool, erdos_renyi: int) -> list(tuple(str, dict)):

    """
    Call the function struc_props with erdos_renyi random graphs with the same number of nodes and edges as G.

    Args:
        G: DiGraph or RegNet.
            Network to compute the structural properties.
        net_id: str.
            Name of the network.
        norm: bool.
            If True, the properties are normalized (biological criteria).
        erdos_renyi: int.
            Number of random graphs to generate with the same number of nodes and edges as G.
            If 0, only the properties of G are computed.
            If greater than 0, the properties of G and the average properties of the random graphs are computed.

    Returns:
        list(tuple(str, dict)): list of tuples with the network id and the properties of the network.

    Raises:
        ValueError: if erdos_renyi is less than 0.
    """


    
    net_id, props = struc_props(G, net_id, norm)

    if erdos_renyi<0:
        raise ValueError('erdos_renyi must be 0 or greater')
    
    elif erdos_renyi==0:
        netid_props = [(net_id, props)]
        
    else:
        
        from collections import defaultdict
        from networkx import fast_gnp_random_graph

        props_er = defaultdict(list)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        for i in range(erdos_renyi):
            ER = nb.RegNet(fast_gnp_random_graph(n, m/(n**2), directed=True))
            _, props = struc_props(ER, f'{net_id}_ER_{i}', norm)
            for k, v in props.items():
                props_er[k].append(v)
            
        props_er_avg = {prop: sum(vals)/len(vals) for prop, vals in props_er.items()}
        netid_props = [(net_id, props), (f'{net_id}_ER_avg', props_er_avg)]

    return netid_props

def save_strucs(
        df: DataFrame,
        output: str=os.getcwd(),
        delimiter: str='\t',
        cl: str=None,
        output_file: str='structural_properties'
        ) -> None:
    
    """
    Save the structural properties in a file.

    Args:
        df: DataFrame.
            Dataframe with the structural properties.
        output: str.
            Path to the output directory. Default is the current working directory.
        delimiter: str.
            Delimiter to use in the output file. Default is tab.
        cl: str.
            Command line used to run the script. Default is None.
        output_file: str.
            Name of the output file. Default is structural_properties.

    Returns:
        None.
    """

    exts = {',' : 'csv', '\t' : 'tsv'}
    ext = exts.get(delimiter, 'txt')
    file_p = concat_path(output, f'{output_file}.{ext}')

    # save output
    # rewrite file if it exists
    with open(file_p, 'w') as f:
        # save command line
        print(cl, file=f)
    
    # save structural properties
    df.to_csv(file_p, sep=delimiter, mode='a')
import os
import pandas as pd
import networkx as nx
import re
import numpy as np

# Auxiliar Fxns
def get_human_names(root, code_names):
    name = re.split('buffer.5000', root)
    name = name[1]
    name = name.replace('\\','')
    return code_names[name]

def extract_networks(directory_name=str, files_ignore=list, cols=tuple, feo=True, code_names=dict):
    networks = {}
    for root, dir, files in os.walk(directory_name):
        if len(files) != 0 and files[0] not in files_ignore:
          path = os.path.join(os.getcwd(), root, files[0])
          net = nx.DiGraph()
          name = get_human_names(root, code_names)
          if feo:
               f = np.genfromtxt(path, dtype= str, delimiter='\t', usecols= cols)
               net.add_edges_from(f)
          else:
               net = nx.read_edgelist(path, create_using= nx.DiGraph)
          
          networks[name] = net
    return networks

if __name__ == '__main__':

    # Biological networks
    hg19CellTypeValues = { 
        "AG10803-DS12374" : "skin fibroblast (abdomen)",
        "AoAF-DS13513" : "aortic adventitial fibroblast",
        "CD20-DS18208" : "B-cells (CD20+)",
        "CD34-DS12274" : "Hematopoietic stem/progenitor cells (CD34+)",
        "fBrain-DS11872" : "fetal brain",
        "fHeart-DS12531" : "fetal heart",
        "fLung-DS14724" : "fetal lung",
        "GM06990-DS7748" : "B-lymphoblastoid cells (individual 06990)",
        "GM12865-DS12436" : "B-lymphoblastoid cells (individual 12865)",
        "HAEpiC-DS12663" : "amniotic epithelium",
        "HAh-DS15192" : "hippocampal astrocyte",
        "HCF-DS12501" : "cardiac fibroblast",
        "HCM-DS12599" : "cardiac myocyte",
        "HCPEpiC-DS12447" : "choroid plexus epithelium",
        "HEEpiC-DS12763" : "esophageal epithelium",
        "HepG2-DS7764" : "hepatic (hepatoblastoma)",
        "hESCT0-DS11909": "embryonic stem cells (line H7)",
        "HFF-DS15115" : "foreskin fibroblast",
        "HIPEpiC-DS12684" : "iris pigment epithelium",
        "HMF-DS13368" : "mammary fibroblast",
        "HMVEC_dBlAd-DS13337" : "microvascular endothelium, adult, blood",
        "HMVEC_dBlNeo-DS13242" : "microvascular endothelium, neonatal, blood",
        "HMVEC_dLyNeo-DS13150" : "microvascular endothelium, neonatal, lympathic",
        "HMVEC_LLy-DS13185" : "microvascular endothelium, lung-derived",
        "HPAF-DS13411" : "pulmonary artery fibroblast",
        "HPdLF-DS13573" : "periodontal ligament fibroblast",
        "HPF-DS13390" : "pulmonary fibroblast",
        "HRCE-DS10666" : "renal cortical epithelium",
        "HSMM-DS14426" : "skeletal muscle myoblast",
        "hTH1-DS7840" : "Th1 T-cells",
        "HVMF-DS13981" : "villous mesenchymal fibroblast",
        "IMR90-DS13219" : "pulmonary fibroblast, fetal",
        "K562-DS9767" : "erythroid (erythroleukemia)",
        "NB4-DS12543" : "granulocytic (promyelocytic leukemia)",
        "NHA-DS12800" : "astrocyte",
        "NHDF_Ad-DS12863" : "dermal fibroblast, adult",
        "NHDF_Neo-DS11923" : "dermal fibroblast, neonatal",
        "NHLF-DS12829" : "lung fibroblast",
        "SAEC-DS10518" : "small airway epithelium",
        "SkMC-DS11949" : "skeletal myocytes",
        "SKNSH-DS8482" : "neuroblastoid (neuroblastoma)"
    }

    human_networks = extract_networks(r'HumanMouseData\human_net_buffer_5000', [], (0,1), feo=False, code_names=hg19CellTypeValues)
    print(len(human_networks))

    # Synthetic networks
    from freyrelab.nets import models

    synth_graphs = {}

    for seed, (net_id, G) in enumerate(human_networks.items()):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        for i in range(1000):
            synth_graphs[f'BA_{net_id}_{i}'] = models.barabasi_albert_graph(n)
            synth_graphs[f'ER_{net_id}_{i}'] = nx.erdos_renyi_graph(n, m/(n*(n-1)), seed=seed, directed=False)

    networks = {**human_networks, **synth_graphs}

    # Graph info
    nets_info = {'Cell Type': [],
            'Nodes': [],
            'Edges': [],
            'Density': []
            }

    for name,G in networks.items():
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = n_edges / (n_nodes * (n_nodes - 1))
        nets_info['Cell Type'].append(name)
        nets_info['Nodes'].append(n_nodes)
        nets_info['Edges'].append(n_edges)
        nets_info['Density'].append(density)

    info_df = pd.DataFrame.from_dict(nets_info)
    path = os.getcwd()
    path += '\human_analysis_nets_info'
    pd.DataFrame.to_csv(info_df, path_or_buf= path, mode='w')

    # Netective Analysis
    import pickle
    from netective import compare_structure

    name_scalars_array, name_moments_array = compare_structure(networks, norm='network', workers=4, return_prop_dicts=True)

    # Save results
    with open('human_analysis_scalars.pkl', 'wb') as f:
        pickle.dump(name_scalars_array, f)

    with open('human_analysis_moments.pkl', 'wb') as f:
        pickle.dump(name_moments_array, f)


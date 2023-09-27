
from tqdm import tqdm
import concurrent.futures
from freyrelab.nets import models, dissimilarity


def run_d(names, nets):
    """
    names = (name1, name2)
    nets = (G1, G2)
    """
    return names, dissimilarity.graph_dissimilarity(*nets)

def run_parallel(f, my_iter, workers):
    len_iter = len(my_iter)
    with tqdm(total=len_iter) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for arg in zip(*my_iter):
                name = arg[0] # tuple (net1, net2)
                # print(f"Running {name}")
                futures[executor.submit(f, *arg)] = name

            results = {}
            for future in concurrent.futures.as_completed(futures):
                try:
                    names, d = future.result()
                    results[names] = d
                    pbar.update(1)
                except Exception as exc:
                    print(f"Error: {exc}")


    return results
from sklearn.metrics.pairwise import *
from multiprocessing import Pool
from workers import wk_dense
from utilities import utils
import numpy as np
import json
import os


def compute_distances():
    """
    Compute pairwise distances of documents as feature vectors.
    
    :return: 
    """

    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    base_labels = utils.get_base_labels()
    uuids = sorted(os.listdir(dir_store))
    words = json.load(open('data/words.json', 'r'))

    for uuid in uuids:
        print(uuid, base_labels[uuid])

    cols = len(words)

    # Force loading of full dataset in RAM (may be a problem with low memory!)
    mini_batch_size = len(uuids)
    decomposed = 0

    file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store))
    pool = Pool(processes=core_num)
    results = pool.map(wk_dense.get_data_matrix, formatted_input)
    pool.close()
    pool.join()

    # sort results
    acc = []
    # Each worker will return a list of size (mini_batch_size / core_num)
    for i in range(core_num):
        for res in results:
            if res[0] == i:
                acc.append(res[1])
    data = np.concatenate(acc)

    print(data.shape)

    metrics = ['cosine', 'euclidean', 'jaccard']
    for metric in metrics:
        try:
            with open("data/distances.txt", "ab") as matrix_file:
                distances = pairwise_distances(data, metric=metric, n_jobs=core_num)
                np.savetxt(matrix_file, distances)
        except:
            print('Exception with ', metric)

if __name__ == '__main__':
    compute_distances()

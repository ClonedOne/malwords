from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_recall_fscore_support
from workers import wk_read_tfidf
from multiprocessing import Pool
from utilities import constants
from sklearn.svm import SVC
from utilities import utils
from scipy.sparse import *
import numpy as np
import json
import os


def classify(config, matrix_file, sparse=False):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return: 
    """

    dir_store = config['dir_store']
    core_num = config['core_num']

    print('Loading data')
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])
    print('Number of distinct families: {}'.format(len(set(base_labels))))

    svc = SVC(kernel='linear')

    scores = cross_val_score(svc, data, base_labels, cv=10, n_jobs=core_num, scoring='f1_micro')

    print('F1 Scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores)/len(scores)))

    if sparse:
        del svc, scores, data
        print('Classifying sparse full-size feature vectors')

        decomposed = 0
        mini_batch_size = len(uuids)
        core_num = config['core_num']
        svc = SVC(kernel='linear')

        print('Loading sparse data')
        data_sparse = get_sparse_data(uuids, decomposed, mini_batch_size, core_num, len(words), words, dir_store)

        scores = cross_val_score(svc, data_sparse, base_labels, cv=10, n_jobs=core_num, scoring='f1_micro')

        print('F1 Scores of cross validation: {}'.format(scores))
        print('Average F1 score: {}'.format(sum(scores)/len(scores)))


def get_sparse_data(uuids, decomposed, mini_batch_size, core_num, cols, words, dir_store):
    """
    Load sparse tf-idf data without dimensionality reduction
    
    :return: 
    """

    # starting from the decomposed-th element of the uuids list
    file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, False))
    pool = Pool(processes=core_num)
    results = pool.map(wk_read_tfidf.get_data_matrix, formatted_input)
    pool.close()
    pool.join()

    # sort results
    acc = []
    for i in range(core_num):
        for res in results:
            if res[0] == i:
                acc.append(res[1])
    data = vstack(acc)

    print(data.shape)
    return data

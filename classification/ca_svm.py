from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from workers import wk_read_tfidf
from sklearn.svm import LinearSVC
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

    print('Loading data')
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Split training and testing data')
    x_train, x_test, y_train, y_test = train_test_split(data, base_labels, test_size=0.2)
    print(len(x_train), len(x_test), len(y_train), len(y_test))

    print('Traing SVM')
    svc = SVC(kernel='linear')
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_test)

    print('Base test labels')
    print(y_test)

    print('SVC prediction')
    print(svc_pred)

    print('Evaluation')
    svc_score = svc.score(x_test, y_test)
    svc_prf = precision_recall_fscore_support(y_test, svc_pred, average='micro')

    print('SVC score:', svc_score)
    print('SVC Precision Recall Fscore: {} {} {}'.format(svc_prf[0], svc_prf[1], svc_prf[2]))

    if sparse:
        del x_train, x_test, y_train, y_test, svc, data, svc_pred

        decomposed = 0
        mini_batch_size = len(uuids)
        core_num = config['core_num']

        data_sparse = get_sparse_data(uuids, decomposed, mini_batch_size, core_num, len(words), words, dir_store)
        x_train, x_test, y_train, y_test = train_test_split(data_sparse, base_labels, test_size=0.2)

        svc = SVC(kernel='linear')
        svc.fit(x_train, y_train)
        svc_pred = svc.predict(x_test)

        svc_score = svc.score(x_test, y_test)
        svc_prf = precision_recall_fscore_support(y_test, svc_pred, average='micro')
        print('SVC score:', svc_score)
        print('SVC Precision Recall Fscore: {} {} {}'.format(svc_prf[0], svc_prf[1], svc_prf[2]))


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

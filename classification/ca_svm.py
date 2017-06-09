from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from helpers import loader_tfidf
from utilities import constants
from sklearn.svm import SVC
import numpy as np
import json
import os


def classify(config, matrix_file, matrix_file_test, uuids, y_train, y_test, sparse=False):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return: 
    """

    dir_store = config['dir_store']
    core_num = config['core_num']

    print('Loading data')
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    data = np.loadtxt(matrix_file)

    print('Training the model')
    svc = SVC(kernel='linear')
    scores = cross_val_score(svc, data, y_train, cv=10, n_jobs=core_num, scoring='f1_micro', verbose=2)

    print('F1 scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores) / len(scores)))

    data = np.loadtxt(matrix_file_test)
    computed_labels = svc.predict(data)

    test_score = f1_score(computed_labels, y_test, 'micro')
    print('F1 score of test: {}'.format(test_score))

    if sparse:
        del svc, scores, data
        print('Classifying sparse full-size feature vectors')

        core_num = config['core_num']
        svc = SVC(kernel='linear')

        print('Loading sparse data')
        data_sparse = loader_tfidf.load_tfidf(uuids, core_num, len(words), words, dir_store, dense=False, ordered=True)

        scores = cross_val_score(svc, data_sparse, y_train, cv=10, n_jobs=core_num, scoring='f1_micro')

        print('F1 Scores of cross validation: {}'.format(scores))
        print('Average F1 score: {}'.format(sum(scores) / len(scores)))

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from visualization import vis_plot
from utilities import interaction
from helpers import loader_tfidf
from utilities import constants
from utilities import output
from sklearn.svm import SVC
import numpy as np
import json
import os


def classify(config, train, test, x_train, x_test, y_train, y_test, sparse=False):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return: 
    """

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    print('Loading data')
    if sparse:
        data_train = loader_tfidf.load_tfidf(x_train, core_num, len(words), words, dir_store, dense=False, ordered=True)
    else:
        data_train = np.loadtxt(train)

    print('10-fold cross validation')
    svc = SVC(kernel='linear')
    scores = cross_val_score(svc, data_train, y_train, cv=10, n_jobs=core_num, scoring='f1_micro', verbose=True)

    print('F1 scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores) / len(scores)))

    print('Training and prediction')
    svc.fit(data_train, y_train)

    if sparse:
        data_test = loader_tfidf.load_tfidf(x_test, core_num, len(words), words, dir_store, dense=False, ordered=True)
    else:
        data_test = np.loadtxt(test)

    computed_labels = svc.predict(data_test)

    test_score = f1_score(y_test, computed_labels, average='micro')
    print('F1 score of test: {}'.format(test_score))

    output.out_classification(dict(zip(x_test, computed_labels.tolist())), 'linear', 'svm')


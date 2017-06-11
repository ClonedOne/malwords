from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import json
import os


def classify(config, train, test, y_train, y_test, sparse=False):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return:
    """

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    dir_store = config['dir_store']
    core_num = config['core_num']
    components = 10000

    print('Loading data')
    if sparse:
        data_train = loader_tfidf.load_tfidf(train, core_num, len(words), words, dir_store, dense=False, ordered=True)
        print('Dimensionality reduction through random projection')
        srp = SparseRandomProjection(n_components=components)
        data_train = srp.fit_transform(data_train)

    else:
        data_train = np.loadtxt(train)

    print('10-fold cross validation')
    mlp = MLPClassifier(
        hidden_layer_sizes=(
            data_train.shape[1],
            int(data_train.shape[1] / 1.5),
            int(data_train.shape[1] / 2),
            int(data_train.shape[1] / 3),
            int(data_train.shape[1] / 4),
            len(set(y_train))
        ),
        max_iter=500,
        solver='adam',
        verbose=True
    )

    scores = cross_val_score(mlp, data_train, y_train, cv=10, n_jobs=core_num, scoring='f1_micro')

    print('F1 Scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores) / len(scores)))

    print('Training and prediction')
    mlp.fit(data_train, y_train)

    if sparse:
        data_test = loader_tfidf.load_tfidf(test, core_num, len(words), words, dir_store, dense=False, ordered=True)
        print('Dimensionality reduction through random projection')
        srp = SparseRandomProjection(n_components=components)
        data_test = srp.fit_transform(data_test)

    else:
        data_test = np.loadtxt(test)

    computed_labels = mlp.predict(data_test)

    test_score = f1_score(computed_labels, y_test, average='micro')
    print('F1 score of test: {}'.format(test_score))

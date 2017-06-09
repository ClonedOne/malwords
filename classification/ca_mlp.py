from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import json
import os


def classify(config, matrix_file, data_matrix_test, uuids, base_labels, sparse=False):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return:
    """

    dir_store = config['dir_store']
    core_num = config['core_num']

    print('Loading data')
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    data = np.loadtxt(matrix_file)

    mlp = MLPClassifier(
        hidden_layer_sizes=(
            data.shape[1],
            int(data.shape[1] * 2),
            int(data.shape[1] * 1.5),
            data.shape[1],
            int(data.shape[1] / 1.5),
            int(data.shape[1] / 2),
            int(data.shape[1] / 3),
            int(data.shape[1] / 4),
            len(set(base_labels))
        ),
        max_iter=20000,
        solver='adam',

    )

    scores = cross_val_score(mlp, data, base_labels, cv=10, n_jobs=core_num, scoring='f1_micro')

    print('F1 Scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores) / len(scores)))

    if sparse:
        del mlp, scores, data
        print('Classifying sparse full-size feature vectors')

        core_num = config['core_num']

        print('Loading sparse data')
        data = loader_tfidf.load_tfidf(uuids, core_num, len(words), words, dir_store, dense=False, ordered=True)

        print('Dimensionality reduction through random projection')
        srp = SparseRandomProjection(n_components=10000)
        data = srp.fit_transform(data)

        print(data.shape)

        mlp = MLPClassifier(
            hidden_layer_sizes=(
                data.shape[1],
                int(data.shape[1] / 1.5),
                int(data.shape[1] / 2),
                int(data.shape[1] / 3),
                int(data.shape[1] / 4),
                len(set(base_labels))
            ),
            max_iter=500,
            solver='adam',

        )

        print('Classification via MLP')
        scores = cross_val_score(mlp, data, base_labels, cv=10, n_jobs=core_num, scoring='f1_micro')

        print('F1 Scores of cross validation: {}'.format(scores))
        print('Average F1 score: {}'.format(sum(scores) / len(scores)))

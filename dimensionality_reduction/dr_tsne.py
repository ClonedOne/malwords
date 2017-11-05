from sklearn.externals import joblib
from sklearn.manifold import TSNE
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import os


def reduce(config, components, uuids=None, x_train=None, x_dev=None, x_test=None):
    """
    Lower dimensionality of data vectors using tSNE.

    :param config: configuration dictionary
    :param components: number of desired components
    :param uuids: list of selected uuids
    :param x_train: List of train set uuids
    :param x_dev: List of dev set uuids
    :param x_test: List of test set uuids
    :return: 
    """

    print('Performing feature extraction using TSNE')

    tsne = TSNE(n_components=components, method='exact', early_exaggeration=6.0, n_iter=1000,
                n_iter_without_progress=100, learning_rate=1000)

    if uuids:
        train = loader_tfidf.load_tfidf(config, uuids, dense=False, ordered=True)
        data = tsne.fit_transform(train)
        rows = len(uuids)
        matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'tsne_{}_{}.txt'.format(components, rows))
        np.savetxt(open(matrix_file, 'wb'), data)

    else:
        t_train = loader_tfidf.load_tfidf(config, x_train, dense=False, ordered=True)
        t_train = tsne.fit_transform(t_train)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'tsne_{}_{}_tr.txt'.format(components, len(t_train))
        )
        np.savetxt(open(matrix_file, 'wb'), t_train)
        rows = len(t_train)

        t_dev = loader_tfidf.load_tfidf(config, x_dev, dense=False, ordered=True)
        t_dev = tsne.fit_transform(t_dev)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'tsne_{}_{}_dv.txt'.format(components, len(t_dev))
        )
        np.savetxt(open(matrix_file, 'wb'), t_dev)

        t_test = loader_tfidf.load_tfidf(config, x_test, dense=False, ordered=True)
        t_test = tsne.fit_transform(t_test)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'tsne_{}_{}_te.txt'.format(components, len(t_test))
        )
        np.savetxt(open(matrix_file, 'wb'), t_test)

        data = (t_train, t_dev, t_test)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'tsne_{}_{}.pkl'.format(components, rows))
    joblib.dump(tsne, model_file)

    return data, tsne

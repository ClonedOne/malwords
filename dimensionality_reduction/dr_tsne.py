from sklearn.externals import joblib
from sklearn.manifold import TSNE
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import os


def reduce(config, uuids, components):
    """
    Lower dimensionality of data vectors using tSNE.

    :param config: configuration dictionary
    :param uuids: list of selected uuids
    :param components: number of desired components
    :return: 
    """

    print('Performing dimensionality reduction using TSNE')

    tsne = TSNE(n_components=components, method='exact', early_exaggeration=6.0, n_iter=1000, metric='cosine',
                n_iter_without_progress=100, learning_rate=1000, verbose=3)

    rows = len(uuids)

    train = loader_tfidf.load_tfidf(config, uuids, dense=False, ordered=True)

    data = tsne.fit_transform(train)

    print('Kullback-Leibler divergence')
    print(tsne.kl_divergence_)

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'tsne_{}_{}.txt'.format(components, rows))
    np.savetxt(open(matrix_file, 'wb'), data)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'tsne_{}_{}.pkl'.format(components, rows))
    joblib.dump(tsne, model_file)

    return data, tsne

from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import os


def reduce(config, uuids, components):
    """
    Lower dimensionality of data vectors using SVD.

    :param config: configuration dictionary
    :param uuids: list of selected uuids
    :param components: number of desired components
    :return: 
    """

    print('Performing dimensionality reduction using SVD')

    svd = TruncatedSVD(n_components=components, n_iter=10)

    rows = len(uuids)

    train = loader_tfidf.load_tfidf(
        config,
        uuids,
        dense=False,
        ordered=False
    )

    data = svd.fit_transform(train)

    print('Explained Variance Ratio')
    print(sum(svd.explained_variance_ratio_))

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'svd_{}_{}.txt'.format(components, rows))
    np.savetxt(open(matrix_file, 'wb'), data)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'svd_{}_{}.pkl'.format(components, rows))
    joblib.dump(svd, model_file)

    return data, svd

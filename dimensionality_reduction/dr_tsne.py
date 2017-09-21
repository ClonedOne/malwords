from sklearn.externals import joblib
from sklearn.manifold import TSNE
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import json
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

    dir_store = config['dir_store']
    core_num = config['core_num']

    tsne = TSNE(n_components=components, method='exact', early_exaggeration=6.0, n_iter=5000, metric='cosine',
                n_iter_without_progress=100, learning_rate=1000, verbose=3)
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    cols = len(words)
    rows = len(uuids)

    # Force loading of full data-set in RAM (may be a problem with low memory!)
    mini_batch_size = rows

    data = transform_vectors(tsne, rows, cols, uuids, words, mini_batch_size, core_num, dir_store)

    print('Kullback-Leibler divergence')
    print(tsne.kl_divergence_)

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'tsne_{}_{}.txt'.format(components, rows))
    np.savetxt(open(matrix_file, 'wb'), data)

    model_file = os.path.join(constants.dir_d, constants.dir_mat, 'tsne_{}_{}.pkl'.format(components, rows))
    joblib.dump(tsne, model_file)

    return data, tsne


def transform_vectors(tsne, rows, cols, uuids, words, mini_batch_size, core_num, dir_store):
    """
    Transform vectors using tSNE    

    :return: 
    """

    decomposed = 0
    new_data = []

    # Divide the documents in mini batches of fixed size and apply Incremental PCA on them
    while decomposed < rows:
        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=False, ordered=True)

        decomposed += mini_batch_size

        new_data.append(tsne.fit_transform(data))

    return np.concatenate(new_data)

from sklearn.metrics.pairwise import pairwise_distances
from distances import jensen_shannon
from helpers import loader_freqs
from utilities import constants
import numpy as np
import json
import os


def get_js(config, uuids):
    """
    Produces a distance matrix applying the Jensen-Shannon Distance to all the feature vectors.
    
    :param config: configuration dictionary
    :param uuids: list of uuids to operate on
    :return:
    """

    core_num = config['core_num']
    dir_malwords = config['dir_mini']

    chosen = False
    while not chosen:
        choice = input(constants.msg_js)

        if choice.lower() == 'y':
            chosen = True

        elif choice.lower() == 'n':
            return

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    cols = len(words)

    print('Creating Jensen-Shannon distance matrix')

    print('Acquiring frequency matrix')
    data = loader_freqs.load_freqs(uuids, core_num, cols, words, dir_malwords, dense=False, ordered=True)
    print(data.shape)
    print(data.sum())

    print('Computing the distances')
    distances = pairwise_distances(data, metric=jensen_shannon.jensen_shannon_dist, n_jobs=core_num)
    matrix_file = os.path.join(constants.dir_d, constants.file_js)
    np.savetxt(matrix_file, distances)

from sklearn.metrics.pairwise import pairwise_distances
from distances import jensen_shannon
from utilities import interaction
from helpers import loader_freqs
from utilities import constants
import numpy as np
import subprocess
import time
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
    dir_malwords = config['dir_malwords']

    file_ext = '_ss.txt'
    zipped = 0
    if os.path.splitext(os.listdir(dir_malwords)[0])[1] == '.gz':
        file_ext = '_ss.txt.gz'
        zipped = 1

    proceed = interaction.ask_yes_no(constants.msg_js)
    if not proceed:
        return

    # words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    # cols = len(words)

    # print('Creating Jensen-Shannon distance matrix')

    # print('Acquiring frequency matrix')
    # data = loader_freqs.load_freqs(uuids, core_num, cols, words, dir_malwords, dense=False, ordered=True)
    # print(data.shape)
    # print(data.sum())

    # print('Computing the distances')
    # distances = pairwise_distances(data, metric=jensen_shannon.jensen_shannon_dist, n_jobs=core_num)
    # matrix_file = os.path.join(constants.dir_d, constants.file_js)
    # np.savetxt(matrix_file, distances)

    uuids_paths = [os.path.join(dir_malwords, uuid + file_ext) for uuid in uuids]
    uuids_paths_file = os.path.join(constants.dir_d, 'uuids_paths.json')

    json.dump(uuids_paths, open(uuids_paths_file, 'w'), indent=2)
    time.sleep(1)

    proc = subprocess.Popen(['./js_dist', str(core_num), uuids_paths_file, str(zipped)])
    proc.wait()

import matplotlib.pyplot as plt
# from utilities import constants
import numpy as np
import json
import os


def take_malfie(uuid_path):
    """
    Generate a picture using the malware features as pixel data.

    :param uuid_path: path to the feature vector file
    :return:
    """

    # words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    words = json.load(open(os.path.join('data/words.json'), 'r'))

    mat_size = 300000
    mat = np.zeros(mat_size)

    tfidfs = json.load(open(uuid_path, 'r'))

    for word, pos in words.items():
        mat[int(pos)] = float(tfidfs.get(word, 0))

    mat = mat.reshape(-1, 500)

    print(mat.shape)
    plt.imshow(mat, cmap="hot")
    plt.show()


if __name__ == '__main__':
    take_malfie('/data/mal_store/000606c3-eecd-4287-943d-eb8e2899e909')

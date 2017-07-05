import numpy as np
import gzip
import os


def get_data_array(data_pack):
    """
    Reads the memhist files and builds up a numpy array with the data.

    :param data_pack: input data for the worker process
    :return: numpy array containing data from memhist files
    """

    process_id = data_pack[0]
    file_list = data_pack[1]
    file_dir = data_pack[2]
    size = data_pack[3]
    mono_bi_tri = data_pack[4]
    positions = data_pack[5]

    n_grams = np.zeros(size, dtype=np.uint64)
    combined = np.zeros(positions[-1], dtype=np.uint64)

    for memhist_file in file_list:
        data = gzip.GzipFile(os.path.join(file_dir, memhist_file)).read()

        if not data:
            continue

        npd = np.frombuffer(data, dtype=np.uint64)
        combined += npd

    n_grams += combined[positions[mono_bi_tri - 1]:positions[mono_bi_tri]]
    n_grams += combined[positions[(mono_bi_tri + 3) - 1]:positions[(mono_bi_tri + 3)]]

    total_ngrams = np.sum(n_grams)

    return process_id, n_grams, total_ngrams

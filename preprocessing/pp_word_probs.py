from workers import wk_read_memhist
from multiprocessing import Pool
from utilities import constants
from utilities import utils
from nltk import trigrams
import numpy as np
import json
import os


def get_word_probabilities():
    # Initialization of variables
    file_dir = "/home/yogaub/projects/projects_data/malrec/memhist"

    # Positions in memhist files
    pos_writes = np.array([0, 2 ** 8, 2 ** 8 + 2 ** 16, 2 ** 8 + 2 ** 16 + 2 ** 24])
    pos_reads = np.array([pos_writes[-1] + i for i in pos_writes[1:]])
    positions = np.concatenate((pos_writes, pos_reads))

    mono_bi_tri = 3
    core_num = 4

    file_list = sorted(os.listdir(file_dir))

    ngram_probs = get_ngrams_probs_multi(file_list, file_dir, mono_bi_tri, core_num, positions)

    ngram_map = get_ngrams_map(ngram_probs, mono_bi_tri, positions)

    word_probs = get_words_probs(ngram_map, os.path.join(constants.dir_d, constants.json_words))

    json.dump(word_probs, open("word_probs.json", "w"), indent=2)

    print(positions)
    print(len(ngram_probs))
    print(len(ngram_map))
    print(len(word_probs))


def get_ngrams_probs_multi(file_list, file_dir, mono_bi_tri, core_num, positions):
    """
    Computes the probability of each n-gram using multiple subprocesses.

    :param positions:
    :param core_num:
    :param file_dir:
    :param file_list: lis of mem-hist files
    :param mono_bi_tri: int specifying monograms/bigrams/trigrams
    :return:
    """

    size = positions[mono_bi_tri] - positions[mono_bi_tri - 1]
    combined = np.zeros(size, dtype=np.uint64)
    total_ngrams = 0.0

    print('Acquiring memhist data')

    file_name_lists = utils.divide_workload(file_list, core_num)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (file_dir, size, mono_bi_tri, positions))
    pool = Pool(processes=core_num)
    results = pool.map(wk_read_memhist.get_data_array, formatted_input)
    pool.close()
    pool.join()

    for result in results:
        total_ngrams += result[2]
        combined += result[1]

    ngram_probs = np.log(combined / total_ngrams)

    return ngram_probs


def get_words_probs(ngram_map, words_file):
    """
    Returns a mapping of words to their probability to be randomly generated.

    :param ngram_map:
    :param words_file:
    :return:
    """

    word_probs = {}
    words = json.load(open(words_file))

    for word in sorted(words):
        trigs = trigrams(word.encode())
        prob = 0.0

        for trig in trigs:
            prob += ngram_map[trig]

        word_probs[word] = prob

    return word_probs


def get_ngrams_map(ngram_probs, mono_bi_tri, positions):
    """
    Returns a mapping of n-gram bytes and their probability.

    :param ngram_probs: probability of each n-gram
    :param mono_bi_tri: int specifying monograms/bigrams/trigrams
    :param positions:
    :return: dictionary mapping bytes and probabilities
    """

    ngram_map = {}
    bswaps = [bswap1, bswap2, bswap3]
    bswap = bswaps[mono_bi_tri - 1]
    displacement = positions[mono_bi_tri - 1]

    for i in range(len(ngram_probs)):
        current_bytes = bswap(i + displacement)
        ngram_map[current_bytes] = ngram_probs[i]

    return ngram_map


def bswap3(i):
    """
    Returns the conversion in byte of a given integer.

    :param i: integer position
    :return:
    """

    return (i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff


def bswap2(i):
    """
    Returns the conversion in byte of a given integer.

    :param i: integer position
    :return:
    """

    return (i >> 8) & 0xff, i & 0xff


def bswap1(i):
    """
    Returns the conversion in byte of a given integer.

    :param i: integer position
    :return:
    """

    return i & 0xff


def out_combined_hist_file(combined):
    """
    Prints out the combined array on file.

    :param combined: Numpy array of unsigned 64 bit integers
    :return:
    """

    with open('combined_histogram.dat', 'w') as memhist_file:
        combined.tofile(memhist_file)


if __name__ == "__main__":
    get_word_probabilities()

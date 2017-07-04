from nltk import trigrams, bigrams, ngrams
# from utilities import constants
import numpy as np
import gzip
import json
import os

file_dir = "/home/yogaub/projects/projects_data/malrec/memhist"

# Positions in memhist files
pos_writes = np.array([0, 2 ** 8, 2 ** 8 + 2 ** 16, 2 ** 8 + 2 ** 16 + 2 ** 24])
pos_reads = np.array([pos_writes[-1] + i for i in pos_writes[1:]])
positions = np.concatenate((pos_writes, pos_reads))
SIZE = pos_reads[-1]


def get_word_probabilities():
    file_list = sorted(os.listdir(file_dir))
    combined = np.zeros(SIZE, dtype=np.uint64)
    ngram_probs = np.zeros(SIZE, dtype=np.float_)
    counters = np.zeros(6, dtype=np.uint64)

    get_ngrams_probs(file_list, combined, ngram_probs, counters)

    ngram_map = get_ngrams_map(ngram_probs)

    # get_words_probs(ngram_map, os.path.join(constants.dir_d, constants.json_words))
    word_probs = get_words_probs(ngram_map, "/home/yogaub/projects/malwords/malwords_cluster/data/words.json")

    json.dump(word_probs, open("word_probs.json", "w"), indent=2)

    print(positions)
    print(counters)
    print(len(ngram_probs))
    print(len(ngram_map))
    print(len(word_probs))


def get_words_probs(ngram_map, words_file):
    """
    Returns a mapping of words to their probability to be randomly generated
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

            if trig not in ngram_map:
                print('{} is not in the map'.format(trig))

            prob += ngram_map[trig]

        word_probs[word] = prob

    return word_probs



def get_ngrams_map(ngram_probs):
    """
    Returns a mapping of n-gram bytes and their probability
    :param ngram_probs: probability of each n-gram
    :return:
    """

    ngram_map = {}
    bswaps = [bswap1, bswap2, bswap3]

    for i in range(len(ngram_probs)):

        for j in range(len(positions)):

            if positions[j] <= i < positions[j + 1]:
                bswap = bswaps[j % 3]
                current_bytes = bswap(i % pos_writes[-1])
                ngram_map[current_bytes] = ngram_probs[i]

    return ngram_map


def get_ngrams_probs(file_list, combined, ngram_probs, counters):
    """
    Computes the probability of each n-gram.
    :param file_list: lis of mem-hist files
    :param combined: Numpy array of unsigned 64 bit integers
    :param ngram_probs: Numpy array to fill up
    :param counters:
    :return:
    """

    for memhist_file in file_list:
        data = gzip.GzipFile(os.path.join(file_dir, memhist_file)).read()

        if not data:
            continue

        npd = np.frombuffer(data, dtype=np.uint64)
        combined += npd

    for i in range(len(counters)):
        counters[i] = np.sum(combined[positions[i]:positions[i + 1]])
        ngram_probs[positions[i]:positions[i + 1]] = np.log(combined[positions[i]:positions[i + 1]] / counters[i])


def bswap3(i):
    """
    Returns the conversion in byte of a given integer
    :param i: integer position
    :return:
    """

    return (i >> 16) & 0xff, (i >> 8) & 0xff, i & 0xff


def bswap2(i):
    """
    Returns the conversion in byte of a given integer
    :param i: integer position
    :return:
    """

    return (i >> 8) & 0xff, i & 0xff


def bswap1(i):
    """
    Returns the conversion in byte of a given integer
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

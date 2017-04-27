from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import *
from sklearn.cluster import KMeans
import numpy as np
import json
import os


dir_store = '/home/yogaub/projects/projects_data/malrec/malwords/store'
num_clusters = 2


def main():
    k_means = KMeans(n_clusters=num_clusters)

    words = json.load(open('data/words.json', 'r'))
    cols = len(words)

    words = get_word_index(words)

    rows = len(os.listdir(dir_store))

    data = lil_matrix((rows, cols))

    uuids = sorted(os.listdir(dir_store))
    row = 0

    # Generates sparse matrix from tf-idf vector files
    for uuid in uuids:
        print('Adding: ' + uuid)

        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    # Convert to sparse matrix format usable by k-menas
    data = data.tocsc()


def extract_tf_idf(tf_idf_file, words):
    """
    Construct an iterator over non zero tf-idf values for each word in the words list
    
    :param tf_idf_file: path to the file containing the bag of words with tf-idf 
    :param words: ordered list of words
    :return: iterator over non zero tf-idf values
    """

    tf_idf_json = json.load(open(tf_idf_file, 'r'))
    for word in tf_idf_json:
        word_index = words[word]
        tf_idf = tf_idf_json[word]
        yield (word_index, tf_idf)


def get_word_index(words):
    """
    Converts word list into dictionary mapping word to index
    
    :param words: 
    :return: 
    """

    word_index = {}
    i = 0

    for word in words:
        word_index[word] = i
        i += 1

    return word_index

if __name__ == '__main__':
    main()

from sklearn import metrics
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

    # Generates sparse matrix from tf-idf vector files
    row = 0
    for uuid in uuids:
        print('Adding: ' + uuid)

        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    # Convert to sparse matrix format usable by k-menas
    print('Converting the sparse matrix')
    data = data.tocsc()

    # apply k-means clustering
    print('Apply KMeans clustering')
    k_means.fit(data)

    computed_labels = k_means.labels_
    print(computed_labels)

    # Retrieve base labels
    base_labels = get_base_labels(uuids)
    base_labels = np.asarray(base_labels)
    print(base_labels)

    # Evaluate clustering
    print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
    print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
    print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
    print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
    print('Completeness:', metrics.completeness_score(base_labels, computed_labels))
    print('Silhouette Coefficient:', metrics.silhouette_samples(data, computed_labels, ))


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


def get_base_labels(uuids):
    """
    Returns the ordered list of base labels from AVClass output
    
    :return: ordered list of labels
    """

    base_labels = []
    uuid_label = json.load(open('data/labels.json'))

    for uuid in uuids:
        base_labels.append(1 if uuid_label[uuid] == 'mydoom' else 0)

    return base_labels


if __name__ == '__main__':
    main()


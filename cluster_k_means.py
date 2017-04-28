from sklearn import metrics
from scipy.sparse import *
from sklearn.cluster import KMeans
import numpy as np
import pickle
import json
import os


dir_store = '/home/yogaub/projects/projects_data/malrec/malwords/store'
num_clusters = 2


def main():
    k_means = KMeans(n_clusters=num_clusters, n_jobs=-1)

    words = json.load(open('data/words.json', 'r'))
    cols = len(words)

    words = get_word_index(words)

    rows = len(os.listdir(dir_store))
    uuids = sorted(os.listdir(dir_store))

    print('Matrix dimensions: ', rows, cols)

    # Retrieve sparese data matrix
    data = get_data_matrix(rows, cols, uuids, words)

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

    result_to_visualize(uuids, base_labels, computed_labels)


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


def get_data_matrix(rows, cols, uuids, words):
    """
    Computes the sparse matrix used as input for the KMeans algorithm.
    
    :param rows: number of rows
    :param cols: number of columns
    :param uuids: list of uuids
    :param words: dictionary of words and their positional index
    :return: sparse tf-idf matrix
    """

    data_matrix_path = 'data/matrix.npz'
    if os.path.isfile(data_matrix_path):
        data = load_sparse_csr(data_matrix_path)
        return data

    data = lil_matrix((rows, cols))

    # Generates sparse matrix from tf-idf vector files
    row = 0
    for uuid in uuids:
        print('Adding: ' + uuid)

        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    # Convert to sparse matrix format usable by k-menas
    data = data.tocsr()
    save_sparse_csr(data_matrix_path, data)

    return data


def save_sparse_csr(filename, array):
    """
    Save sparse matrix to file
    
    :param filename: name of the data matrix file
    :param array: sparse matrix
    :return: 
    """

    print('Saving matrix to file')
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    """
    Loads sparse matrix from file 
    
    :param filename: name of the data matrix file
    :return: 
    """

    print('Loading matrix from file')
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def result_to_visualize(uuids, base_labels, computed_labels):
    """
    Generate a json file structured so it can be used for visualization 
    
    :param uuids: list of uuids
    :param base_labels: base truth labels
    :param computed_labels: clustering results 
    :return: 
    """

    out_dict = {'name': 'clustering', 'children': []}

    for i in range(num_clusters):
        child_dict = {'name': str(i), 'children': []}

        for j in range(len(computed_labels)):
            label = int(computed_labels[j])
            if label == i:
                true_label = int(base_labels[j])
                child_inner = {'name': uuids[j], 'color': true_label}
                child_dict['children'].append(child_inner)

        out_dict['children'].append(child_dict)

    json.dump(out_dict, open('visualize/graph1.json', 'w'))


if __name__ == '__main__':
    main()


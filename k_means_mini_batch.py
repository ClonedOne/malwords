from sklearn.cluster import MiniBatchKMeans
from multiprocessing import Pool
from sklearn import metrics
from utilities import utils
from scipy.sparse import *
import numpy as np
import random
import json
import os

dir_store = ''
num_clusters = 6
mini_batch_size = 300
core_num = 4


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 
    
    :return: 
    """

    global dir_store
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    k_means = MiniBatchKMeans(n_clusters=num_clusters, batch_size=mini_batch_size)
    words = json.load(open('data/words.json', 'r'))
    uuids = sorted(os.listdir(dir_store))
    rand_uuids = random.sample(uuids, len(uuids))

    clustered = 0
    cols = len(words)
    rows = len(uuids)

    print('Matrix dimensions: ', rows, cols)

    # Retrieve base labels
    base_labels = get_base_labels(uuids)
    base_labels = np.asarray(base_labels)
    print('Base labels')
    print(base_labels)

    print('\nTraining KMeans')
    train_k_means(k_means, clustered, rows, cols, rand_uuids, words)
    clustered = 0

    print('\nPredicting values')
    computed_labels = apply_k_means(k_means, clustered, rows, cols, uuids, words)
    print(computed_labels.tolist())

    # Evaluate clustering
    print('Clustering evaluation')
    print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
    print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
    print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
    print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
    print('Completeness:', metrics.completeness_score(base_labels, computed_labels))

    result_to_visualize(uuids, base_labels, computed_labels)


def train_k_means(k_means, clustered, rows, cols, rand_uuids, words):
    """
    Train the MiniBatchKMeans clustering algorithm with fixed size data batches taken random from the input data-set. 
    
    :param k_means: MiniBatchKMeans object
    :param clustered: counter
    :param rows: number of rows of the data set matrix
    :param cols: number of columns of the data set matrix
    :param rand_uuids: list of documents in randomized order
    :param words: dictionary mapping word features to their index
    :return: 
    """

    # Divide the docuements in mini batches of fixed size and train the KMeans
    while clustered < rows:
        print('Processing documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        # Each worker will receive a list of size (mini_batch_size / core_num)
        # starting from the clustered-th element of the uuids list
        file_name_lists = utils.divide_workload(rand_uuids[clustered:][:mini_batch_size], core_num)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words))
        pool = Pool(processes=core_num)
        results = pool.map(get_data_matrix, formatted_input)
        pool.close()
        pool.join()

        # sort results
        acc = []
        for i in range(core_num):
            for res in results:
                if res[0] == i:
                    acc.append(res[1])
        data = vstack(acc)

        del results
        del acc
        clustered += mini_batch_size

        # train k-means on the mini batch
        print('Train KMeans with mini batch')
        k_means.partial_fit(data)


def apply_k_means(k_means, clustered, rows, cols, uuids, words):
    """
    Apply the MiniBatchKMeans clustering algorithm with fixed size data batches taken in order from the input data set. 

    :param k_means: MiniBatchKMeans object
    :param clustered: counter
    :param rows: number of rows of the data set matrix
    :param cols: number of columns of the data set matrix
    :param uuids: list of documents in order
    :param words: dictionary mapping word features to their index
    :return: labels computed by KMeans over the data set
    """

    computed_labels = np.array([])

    # Divide the docuements in mini batches of fixed size and apply KMeans on them
    while clustered < rows:
        print('Processing documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        # Each worker will receive a list of size (mini_batch_size / core_num)
        # starting from the clustered-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[clustered:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words))
        pool = Pool(processes=core_num)
        results = pool.map(get_data_matrix, formatted_input)
        pool.close()
        pool.join()

        # sort results
        acc = []
        for i in range(core_num):
            for res in results:
                if res[0] == i:
                    acc.append(res[1])
        data = vstack(acc)

        del results
        del acc
        clustered += mini_batch_size

        # train k-means on the mini batch
        print('Apply KMeans on mini batch')
        batch_computed_labels = k_means.predict(data)

        computed_labels = np.append(computed_labels, batch_computed_labels)

    return computed_labels


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


def get_base_labels(uuids):
    """
    Returns the ordered list of base labels from AVClass output

    :return: ordered list of labels
    """

    base_labels = []
    uuid_label = json.load(open('data/labels.json'))
    families = {'mydoom': 0,
                'neobar': 1,
                'gepys': 2,
                'lamer': 3,
                'neshta': 4,
                'bladabindi': 5
                }

    for uuid in uuids:
        base_labels.append(families[uuid_label[uuid]])

    return base_labels


def get_data_matrix(data_pack):
    """
    Computes the sparse matrix used as input for the KMeans algorithm.
    The data pack contains:

     * number of rows
     * number of columns
     * list of uuids
     * dictionary of words and their positional index
    
    :param data_pack: input data for the worker process
    :return: sparse tf-idf matrix
    """

    # Unpacking data from main process
    process_id = data_pack[0]
    uuids = data_pack[1]
    rows = len(uuids)
    cols = data_pack[2]
    words = data_pack[3]

    # print(len(uuids), rows, cols, len(words))

    data = lil_matrix((rows, cols))

    # Generates sparse matrix from tf-idf vector files
    row = 0
    for uuid in uuids:
        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    # Convert to sparse matrix format usable by k-menas
    data = data.tocoo()
    print('{} - {}'.format(process_id, data.count_nonzero()))
    return process_id, data


#
# Visualization
#

def result_to_visualize(uuids, base_labels, computed_labels):
    """
    Generate a json file structured so it can be used for visualization 

    :param uuids: list of uuids
    :param base_labels: base truth labels
    :param computed_labels: clustering results 
    :return: 
    """

    out_dict = {'name': 'clustering', 'children': []}
    colors = {0: 'blue',
              1: 'yellow',
              2: 'red',
              3: 'green',
              4: 'orange',
              5: 'brown'
              }

    for i in range(num_clusters):
        child_dict = {'name': str(i), 'children': []}

        for j in range(len(computed_labels)):
            label = int(computed_labels[j])
            if label == i:
                true_label = int(base_labels[j])
                child_inner = {'name': uuids[j], 'color': colors[true_label]}
                child_dict['children'].append(child_inner)

        out_dict['children'].append(child_dict)

    graph_path = 'visualize/graph1.json'
    json.dump(out_dict, open(graph_path, 'w'), indent=2)


if __name__ == '__main__':
    cluster()

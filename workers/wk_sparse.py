from scipy.sparse import *
import json
import os


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


def get_data_matrix(data_pack):
    """
    Computes the sparse matrix used as input for algorithm.
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
    dir_store = data_pack[4]

    # print(len(uuids), rows, cols, len(words))

    data = lil_matrix((rows, cols))

    # Generates sparse matrix from tf-idf vector files
    row = 0
    for uuid in uuids:
        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    # Convert to coo sparse matrix format
    data = data.tocoo()
    print('{} - {} - {}'.format(process_id, data.count_nonzero(), data.data.nbytes))
    return process_id, data

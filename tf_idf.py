from multiprocessing import Pool
from collections import Counter
from utilities import utils
import json
import math
import os

dir_malwords = ''
dir_store = ''
core_num = 4


def get_tf_idf():
    """
    Compute the tf-idf value of each word of each document
    
    :return: 
    """
    global dir_malwords, dir_store
    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
    total_documents = float(len(os.listdir(dir_malwords)))

    print('Acquiring document frequencies')
    file_name_lists = utils.divide_workload(sorted(os.listdir(dir_malwords)), core_num)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (total_documents, ))
    pool = Pool(processes=core_num)
    results = pool.map(compute_df, formatted_input)
    pool.close()
    pool.join()
    dfs = Counter()
    for sub_dfs in results:
        dfs += sub_dfs
    json.dump(dfs, open('data/dfs.json', 'w'), indent=2)

    print('Lowering features dimensionality')
    remove_useless_words(dfs, total_documents, 0.75)

    print('Computing Tf-Idf values')
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (dfs, total_documents))
    pool = Pool(processes=core_num)
    pool.map(compute_tf_idf, formatted_input)
    pool.close()
    pool.join()


def compute_df(data_pack):
    """
    Scans the bag-of-words documents and computes the document frequency of each word.

    :return: Counter containing the document frequency of each word
    """

    file_name_list = data_pack[1]
    dfs = Counter()

    for sample in sorted(file_name_list):

        with open(os.path.join(dir_malwords, sample), 'rb') as words_file:

            for line in words_file:
                line = line.strip().split()

                word = line[0].decode('utf-8')
                dfs[word] += 1

    return dfs


def remove_useless_words(dfs, total_documents, filter_factor):
    """
    Remove words if:
     * they appear in all documents (would end up with tf-idf = 0)
     * they appear just once in all the documents
     * their document frequency is above a threshold (simulate stopwords elimination)

    :param dfs: document frequency of each word
    :param total_documents: number of documents
    :param filter_factor: filtering factor
    :return: 
    """

    to_remove = set()
    threshold = filter_factor * total_documents

    print('Initial features size:', len(dfs))

    for word in dfs:
        if dfs[word] == total_documents or dfs[word] == 1 or dfs[word] >= threshold:
            to_remove.add(word)

    for word in to_remove:
        dfs.pop(word, None)

    print('Features size:', len(dfs))

    # Create a dictionary mapping each (sorted) word with a numerical index
    words = dict(zip(sorted(list(dfs.keys())), list(range(len(dfs)))))

    json.dump(words, open('data/words.json', 'w'), indent=2)


def compute_tf_idf(data_pack):
    """
    Scans the bag-of-words documents and computes the term frequency value of each word.
    During the process computes the document frequency of each word.

    :return:  
    """

    file_name_list = data_pack[1]
    dfs = data_pack[2]
    total_documents = data_pack[3]

    norm_factor = 0.4

    for sample in sorted(file_name_list):
        words = Counter()
        total_words = 0
        document_length = 0

        # Scan once the bag of words files and memorize the words in a temporary per-file dictionary
        with open(os.path.join(dir_malwords, sample), 'rb') as words_file:
            for line in words_file:
                line = line.strip().split()

                word = line[0].decode('utf-8')
                count = int(line[1])

                # avoid words which would end up with tf-idf = 0
                if word not in dfs:
                    continue

                words[word] = count
                document_length += count
                total_words += 1

        uuid = sample[:-7]
        tf_idf = {}

        # find the most frequent word
        most_freq = max(list(words.values()))

        # Compute the term frequency of a word using double normalization
        for word in words:
            tf = norm_factor + ((1 - norm_factor) * (float(words[word]) / float(most_freq)))
            idf = math.log(total_documents / float(dfs[word]))
            tf_idf[word] = tf * idf

        json.dump(tf_idf, open(os.path.join(dir_store, uuid), "w"), indent=2)
        print(uuid, total_words, document_length)


if __name__ == '__main__':
    get_tf_idf()

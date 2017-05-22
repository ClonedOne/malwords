from multiprocessing import Pool
from utilities import constants
from collections import Counter
from workers import wk_tfidf
from utilities import utils
import json
import os


def get_idf(config):
    """
    Computes the inverted document frequency for each word in the data-set and removes redundant features. 
    
    :return: 
    """

    dir_malwords = config['dir_mini']
    core_num = config['core_num']
    dir_base = config['dir_base']

    total_documents = float(len(os.listdir(dir_malwords)))

    print('Acquiring document frequencies')
    file_name_lists = utils.divide_workload(sorted(os.listdir(dir_malwords)), core_num)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (dir_malwords,))
    pool = Pool(processes=core_num)
    results = pool.map(wk_tfidf.compute_df, formatted_input)
    pool.close()
    pool.join()

    dfs = Counter()
    for sub_dfs in results:
        dfs += sub_dfs

    json.dump(dfs, open(os.path.join(constants.dir_d, constants.json_dfs), 'w'), indent=2)

    print('Lowering features dimensionality')
    remove_useless_words(dfs, total_documents, 0.5, 0.005, dir_base)


def remove_useless_words(dfs, total_documents, filter_high, filter_low, dir_base):
    """
    Remove words if:
     * they appear in all documents (would end up with tf-idf = 0)
     * they appear just once in all the documents
     * their document frequency is above a threshold_high (simulate stopwords elimination)

    :param dfs: document frequency of each word
    :param total_documents: number of documents
    :param filter_high: filtering factor
    :param filter_low: filtering factor
    :param dir_base: directory containing the base words file
    :return: 
    """

    to_remove = set()
    threshold_high = int(filter_high * total_documents)
    threshold_low = int(filter_low * total_documents)

    singleton_words = 0
    frequent_words = 0
    rare_words = 0
    in_base = 0
    digits = 0

    print('Initial features number:', len(dfs))
    print('Document frequency thresholds: {} {}'.format(threshold_low, threshold_high))

    base_words = utils.get_base_words(dir_base)

    for word in dfs:

        if word.isdigit():
            to_remove.add(word)
            digits += 1

        elif dfs[word] == 1:
            to_remove.add(word)
            singleton_words += 1

        elif dfs[word] >= threshold_high:
            to_remove.add(word)
            frequent_words += 1

        elif dfs[word] <= threshold_low:
            to_remove.add(word)
            rare_words += 1

        elif word in base_words:
            to_remove.add(word)
            in_base += 1

    for word in to_remove:
        dfs.pop(word, None)

    print('Features number:', len(dfs))
    print('Words containing only digits:', digits)
    print('Words appearing only once:', singleton_words)
    print('Words over threshold_high frequency:', frequent_words)
    print('Words under threshold_low frequency:', rare_words)
    print('Words appearing in base:', in_base)

    # Create a dictionary mapping each (sorted) word with a numerical index
    words = dict(zip(sorted(list(dfs.keys())), list(range(len(dfs)))))

    json.dump(words, open(os.path.join(constants.dir_d, constants.json_words), 'w'), indent=2)

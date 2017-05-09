from multiprocessing import Pool
from collections import Counter
from workers import wk_tfidf
from utilities import utils
import json
import os

dir_malwords = ''
dir_store = ''
dir_base = ''
core_num = 1


def get_tf_idf():
    """
    Compute the tf-idf value of each word of each document
    
    :return: 
    """

    global dir_malwords, dir_store, dir_base, core_num

    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
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
    json.dump(dfs, open('data/dfs.json', 'w'), indent=2)

    print('Lowering features dimensionality')
    remove_useless_words(dfs, total_documents, 0.5)

    print('Computing Tf-Idf values')
    formatted_input = utils.format_worker_input(core_num, file_name_lists,
                                                (dfs, total_documents, dir_malwords, dir_store))
    pool = Pool(processes=core_num)
    pool.map(wk_tfidf.compute_tf_idf, formatted_input)
    pool.close()
    pool.join()


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

    singleton_words = 0
    frequent_words = 0
    in_base = 0

    print('Initial features number:', len(dfs))
    print('Document frequency threshold:', threshold)

    base_words = set()
    base_file = os.path.join(dir_base, 'base.txt')
    if os.path.isfile(base_file):
        with open(base_file, 'rb') as base_in:
            for line in base_in:
                base_words.add(line.strip().split()[0].decode('utf-8'))

    for word in dfs:
        if dfs[word] == 1:
            to_remove.add(word)
            singleton_words += 1
        if dfs[word] >= threshold:
            to_remove.add(word)
            frequent_words += 1
        if word in base_words:
            to_remove.add(word)
            in_base += 1

    for word in to_remove:
        dfs.pop(word, None)

    print('Features number:', len(dfs))
    print('Words appearing only once:', singleton_words)
    print('Words with over threshold frequency:', frequent_words)
    print('Words appearing in base:', in_base)

    # Create a dictionary mapping each (sorted) word with a numerical index
    words = dict(zip(sorted(list(dfs.keys())), list(range(len(dfs)))))

    json.dump(words, open('data/words.json', 'w'), indent=2)


if __name__ == '__main__':
    get_tf_idf()

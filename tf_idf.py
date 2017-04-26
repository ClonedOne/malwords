from collections import Counter
from pprint import pprint
import pickle
import json
import math
import gzip
import os

dir_malwords = '/home/yogaub/projects/projects_data/malrec/malwords/mini_malwords'
dir_store = '/home/yogaub/projects/projects_data/malrec/malwords/store'


def get_tf_idf():
    total_documents = len(os.listdir(dir_malwords))
    dfs = compute_df()
    remove_useless_words(dfs, total_documents)
    compute_tf(dfs)
    compute_tf_idf(dfs, total_documents)


def compute_df():
    """
    Scans the bag-of-words documents and computes the document frequency of each word.
    
    :return: Counter containing the document frequency of each word
    """

    dfs = Counter()

    for sample in sorted(os.listdir(dir_malwords)):

        with gzip.open(os.path.join(dir_malwords, sample), 'r') as words_file:

            for line in words_file:
                line = line.strip().split()

                word = line[0].decode('utf-8')
                dfs[word] += 1

    return dfs


def remove_useless_words(dfs, total_documents):
    """
    Remove words which would end up with tf-idf = 0
    
    :param dfs: document frequency of each word
    :param total_documents: number of documents
    :return: 
    """

    to_remove = set()

    for word in dfs:
        if dfs[word] == total_documents:
            to_remove.add(word)

    for word in to_remove:
        dfs.pop(word, None)


def compute_tf(dfs):
    """
    Scans the bag-of-words documents and computes the term frequency value of each word.
    During the process computes the document frequency of each word.
    
    :return:  
    """

    norm_factor = 0.4

    for sample in sorted(os.listdir(dir_malwords)):
        words = Counter()
        total_words = 0
        document_length = 0

        # Scan once the bag of words files and memorize the words in a temporary per-file dictionary
        with gzip.open(os.path.join(dir_malwords, sample), 'r') as words_file:
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

        uuid = sample[:-10]
        tfs = {}

        # find the most frequent word
        most_freq = max(list(words.values()))

        # Compute the term frequency of a word using double normalization
        for word in words:
            tf = norm_factor + ((1 - norm_factor) * (float(words[word]) / float(most_freq)))
            tfs[word] = tf

        pickle.dump(tfs, open(os.path.join(dir_store, uuid), "wb"))
        print(uuid, total_words, document_length)


def compute_tf_idf(dfs, total_documents):
    """
    Scans the pickled files and computes the tf-idf value of each word.
    
    :param dfs: Counter containing the document frequency of each word
    :param total_documents: number of documents
    :return: 
    """

    for uuid in sorted(os.listdir(dir_store)):
        word_tf_idf = {}
        tfs = pickle.load(open(os.path.join(dir_store, uuid), 'rb'))

        for word in tfs:

            idf = math.log(float(total_documents) / float(dfs[word]))
            tf_idf = tfs[word] * idf
            word_tf_idf[word] = tf_idf

        json.dump(word_tf_idf, open(os.path.join(dir_store, uuid), "w"), indent=2)


if __name__ == '__main__':
    get_tf_idf()

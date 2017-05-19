from collections import Counter
import json
import math
import os


def compute_tf_idf(data_pack):
    """
    Scans the bag-of-words documents and computes the term frequency value of each word.
    During the process computes the document frequency of each word.

    :return:  
    """

    file_name_list = data_pack[1]
    dfs = data_pack[2]
    total_documents = data_pack[3]
    dir_malwords = data_pack[4]
    dir_store = data_pack[5]
    multiplier = data_pack[6]

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

                # avoid excluded words
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
            if multiplier:
                tf = tf * len(word)
            idf = math.log(total_documents / float(dfs[word]))
            tf_idf[word] = tf * idf

        json.dump(tf_idf, open(os.path.join(dir_store, uuid), "w"), indent=2)
        # json.dump(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True), open(os.path.join('temp', uuid), "w"), indent=2)
        # print(uuid, total_words, document_length)


def compute_df(data_pack):
    """
    Scans the bag-of-words documents and computes the document frequency of each word.

    :return: Counter containing the document frequency of each word
    """

    file_name_list = data_pack[1]
    dir_malwords = data_pack[2]
    dfs = Counter()

    for sample in sorted(file_name_list):

        with open(os.path.join(dir_malwords, sample), 'rb') as words_file:

            for line in words_file:
                line = line.strip().split()

                word = line[0].decode('utf-8')
                dfs[word] += 1

    return dfs

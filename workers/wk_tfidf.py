from collections import Counter
import subprocess
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
    words_probs = data_pack[6]
    selected_words = data_pack[7]

    norm_factor = 0.4

    for sample in sorted(file_name_list):

        # Initialize per-file variables
        freqs_file = os.path.join(dir_malwords, sample)
        uuid = sample.split('.')[0][:-3]
        words = Counter()
        in_file = None
        proc = None
        tf_idf = {}

        # If it is a zipped file use fast gzip cat
        if os.path.splitext(freqs_file)[1] == '.gz':
            proc = subprocess.Popen(['gzip', '-cdfq', freqs_file], stdout=subprocess.PIPE, bufsize=4096)
            lines = proc.stdout
        else:
            in_file = open(freqs_file, 'rb')
            lines = in_file

        # Retrieve word frequencies
        for line in lines:
            line = line.strip().split()

            word = line[0].decode('utf-8')
            count = int(line[1])

            # avoid excluded words
            if word not in selected_words:
                continue

            words[word] = count

        # Cleanup
        if os.path.splitext(freqs_file)[1] == '.gz':
            proc.terminate()
        else:
            in_file.close()

        # find the most frequent word
        most_freq = max(list(words.values()))

        # Compute the term frequency of a word using double normalization
        for word in words:
            tf = norm_factor + ((1 - norm_factor) * (float(words[word]) / float(most_freq)))

            if not words_probs:
                tf = tf * len(word)
            else:
                tf = tf * (-1) * words_probs[word] / len(word)

            idf = math.log(total_documents / float(dfs[word]))
            tf_idf[word] = tf * idf

        json.dump(tf_idf, open(os.path.join(dir_store, uuid), "w"), indent=2)


def compute_df(data_pack):
    """
    Scans the bag-of-words documents and computes the document frequency of each word.

    :return: Counter containing the document frequency of each word
    """

    file_name_list = data_pack[1]
    dir_malwords = data_pack[2]
    dfs = Counter()

    for sample in sorted(file_name_list):

        # Initialize per-file variables
        freqs_file = os.path.join(dir_malwords, sample)
        in_file = None
        proc = None

        # If it is a zipped file use fast gzip cat
        if os.path.splitext(freqs_file)[1] == '.gz':
            proc = subprocess.Popen(['gzip', '-cdfq', freqs_file], stdout=subprocess.PIPE, bufsize=4096)
            lines = proc.stdout
        else:
            in_file = open(freqs_file, 'rb')
            lines = in_file

        for line in lines:
            line = line.strip().split()

            word = line[0].decode('utf-8')
            dfs[word] += 1

        # Cleanup
        if os.path.splitext(freqs_file)[1] == '.gz':
            proc.terminate()
        else:
            in_file.close()

    return dfs

from scipy.sparse import *
import numpy as np
import subprocess
import os


def get_data_matrix(data_pack):
    """
    Reads word frequencies from bag-of-words files and returns them in a sparse or dense matrix. 
    The data pack contains:
    
     * process id
     * list of uuids
     * number of columns
     * dictionary of words and their positional index
     * directory where files are stored
     * flag, if set the matrix must be dense
    
    :param data_pack: input data for the worker process
    :return: dense or sparse frequency matrix
    """

    # Unpacking data from main process
    process_id = data_pack[0]
    uuids = data_pack[1]
    rows = len(uuids)
    cols = data_pack[2]
    words = data_pack[3]
    dir_files = data_pack[4]
    dense = data_pack[5]

    if dense:
        data = np.zeros((rows, cols))
    else:
        data = lil_matrix((rows, cols))

    row = 0
    for uuid in uuids:
        cur_row = extract_freqs(os.path.join(dir_files, uuid + '_ss.txt'), words, cols)
        data[row] = cur_row
        row += 1

    if dense:
        print(data.nbytes)
    else:
        print(data.data.nbytes)
    return process_id, data


def extract_freqs(freqs_file, words, cols):
    """
    Extracts frequency vectors from bag-of-words files. Supports gzipped files.
    
    :param freqs_file: bag-of-words file 
    :param words: dictionary of valid words and related indices
    :param cols: number of features of the vector
    :return: frequency vector
    """

    cur_row = np.zeros(cols)
    proc = None
    in_file = None

    if os.path.splitext(freqs_file)[1] == '.gz':
        proc = subprocess.Popen(['gzip', '-cdfq', freqs_file], stdout=subprocess.PIPE, bufsize=4096)
        lines = proc.stdout
    else:
        in_file = open(freqs_file, 'rb')
        lines = in_file

    for line in lines:
        line = line.strip().split()

        word = line[0].decode('utf-8')
        count = int(line[1])

        if word not in words:
            continue

        word_id = words[word]
        cur_row[word_id] = count

    if os.path.splitext(freqs_file)[1] == '.gz':
        proc.terminate()
    else:
        in_file.close()

    return cur_row

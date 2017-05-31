from multiprocessing import Pool
from utilities import constants
from workers import wk_tfidf
from utilities import utils
import json
import os


def get_tf_idf(config):
    """
    Compute the tf-idf value of each word of each document
    
    :return: 
    """

    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
    core_num = config['core_num']

    total_documents = float(len(os.listdir(dir_malwords)))

    dfs = json.load(open(os.path.join(constants.dir_d, constants.json_dfs)))
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words)))

    file_name_lists = utils.divide_workload(sorted(os.listdir(dir_malwords)), core_num)

    print('Computing Tf-Idf values')
    formatted_input = utils.format_worker_input(core_num, file_name_lists,
                                                (dfs, total_documents, dir_malwords, dir_store, True, words))
    pool = Pool(processes=core_num)
    pool.map(wk_tfidf.compute_tf_idf, formatted_input)
    pool.close()
    pool.join()

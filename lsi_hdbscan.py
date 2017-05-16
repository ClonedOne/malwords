from utilities import utils, evaluation, output
from gensim import corpora, models, matutils
import numpy as np
import hdbscan
import json
import os

dir_malwords = ''
dir_store = ''
dir_base = ''
core_num = 1


def cluster():
    """
    Retrieve topics in the documents and cluster them using topics as features.

    :return: 
    """

    global dir_malwords, dir_store, dir_base, core_num

    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
    core_num = config['core_num']
    dir_base = config['dir_base']

    words = json.load(open('data/words.json', 'r'))
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Retrieving corpus')
    malw_corpus = corpora.MmCorpus('data_gensim/corpus.mm')

    inv_words = {v: k for k, v in words.items()}
    dictionary = corpora.Dictionary.from_corpus(malw_corpus, id2word=inv_words)

    print('Computing Tf-Idf')
    tfidf = models.TfidfModel(malw_corpus)
    tfidf_corpus = tfidf[malw_corpus]
    print(len(tfidf_corpus))

    print('Computing LSI')
    n_topics = 400
    lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=n_topics)
    lsi_corpus = lsi[tfidf_corpus]
    print(len(lsi_corpus))

    data = matutils.corpus2dense(lsi_corpus, num_terms=n_topics, num_docs=len(uuids)).T
    print(data.shape)

    print('Perform clustering with euclidean distance')
    hdbs = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean', match_reference_implementation=True)
    hdbs.fit(data)
    computed_labels = hdbs.labels_
    num_clusters = len(set(computed_labels))

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'euclidean', 'hdbscan')


if __name__ == '__main__':
    cluster()

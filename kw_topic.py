from gensim import corpora, models
import json


dir_malwords = ''
dir_store = ''
dir_base = ''
core_num = 1


def get_topics():
    """
    Retrieve topics in the documents clustering.
    
    :return: 
    """

    global dir_malwords, dir_store, dir_base, core_num

    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
    core_num = config['core_num']
    dir_base = config['dir_base']

    words = json.load(open('data/words.json', 'r'))

    malw_corpus = corpora.MmCorpus('gensim/corpus.mm')
    inv_words = {v: k for k, v in words.items()}
    dictionary = corpora.Dictionary.from_corpus(malw_corpus, id2word=inv_words)
    print(dictionary)

    tfidf = models.TfidfModel(malw_corpus)
    tfidf_corpus = tfidf[malw_corpus]

    # n_topics = 7
    # lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=n_topics)
    # # lsi_corpus = lsi[tfidf_corpus]
    #
    # print('LSI')
    # for topic in lsi.print_topics(n_topics):
    #     print(topic)
    # print('\n')
    #
    # del lsi
    #
    # print('LDA')
    #
    # lda = models.LdaModel(tfidf_corpus, id2word=dictionary, num_topics=n_topics)
    #
    # for topic in lda.print_topics(n_topics):
    #     print(topic)
    # print('\n')
    #
    # del lda
    #
    # lda = models.LdaMulticore(tfidf_corpus, id2word=dictionary, num_topics=n_topics, workers=core_num)
    #
    # for topic in lda.print_topics(n_topics):
    #     print(topic)
    # print('\n')
    #
    # del lda
    #
    # print('HDP')
    #
    hdp = models.HdpModel(tfidf_corpus, id2word=dictionary)

    for topic in hdp.print_topics():
        print(topic)
    print('\n')


if __name__ == '__main__':
    get_topics()


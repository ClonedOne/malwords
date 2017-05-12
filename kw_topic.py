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

    lsi = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=4)
    lsi_corpus = lsi[tfidf_corpus]

    for doc in lsi_corpus:
        print(doc)

    for topic in lsi.print_topics(4):
        print(topic)

if __name__ == '__main__':
    get_topics()

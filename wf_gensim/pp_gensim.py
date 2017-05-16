from gensim import corpora
import json
import os

dir_malwords = ''
words = {}
dictionary = None


class MyCorpus(object):

    def __init__(self, uuids=None):
        if uuids:
            self.uuids = uuids
        else:
            self.uuids = None

    def __iter__(self):

        if not self.uuids:
            self.uuids = sorted(os.listdir(dir_malwords))
        else:
            self.uuids = sorted([uuid + '_ss.txt' for uuid in self.uuids])

        for bow_file in self.uuids:

            word_vec = []

            for line in open(os.path.join(dir_malwords, bow_file), 'rb'):
                word, count = line.split()
                word = word.decode('utf-8')
                count = int(count)

                if word not in words:
                    continue

                word_id = words[word]

                word_vec.append((word_id, count))

            yield word_vec


def prepare_corpus(uuids=None):
    """
    Generates a gensim corpus object from the bag of words.
    
    :param uuids: List of uuids
    :return: 
    """

    global dir_malwords, words, dictionary

    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    words = json.load(open('data/words.json', 'r'))

    corpus_dir = 'data_gensim'

    if not os.path.exists(corpus_dir):
        os.mkdir(corpus_dir)

    corpus = MyCorpus(uuids)

    corpora.MmCorpus.serialize(os.path.join(corpus_dir, 'corpus.mm'), corpus)

    new_corpus = corpora.MmCorpus(os.path.join(corpus_dir, 'corpus.mm'))
    inv_words = {v: k for k, v in words.items()}
    dictionary = corpora.Dictionary.from_corpus(new_corpus, id2word=inv_words)

    print(type(new_corpus))
    print(type(dictionary))


if __name__ == '__main__':
    prepare_corpus()

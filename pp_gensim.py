from gensim import corpora
import json
import os

dir_malwords = ''
words = {}
dictionary = None


class MyCorpus(object):

    def __iter__(self):

        for bow_file in sorted(os.listdir(dir_malwords)):

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


def prepare_corpus():
    """
    Generates a gensim corpus object from the bag of words.
    
    :return: 
    """

    global dir_malwords, words, dictionary

    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']

    words = json.load(open('data/words.json', 'r'))

    corpus = MyCorpus()

    corpora.MmCorpus.serialize('gensim/corpus.mm', corpus)

    new_corpus = corpora.MmCorpus('gensim/corpus.mm')
    inv_words = {v: k for k, v in words.items()}
    dictionary = corpora.Dictionary.from_corpus(new_corpus, id2word=inv_words)

    print(type(new_corpus))
    print(type(dictionary))


if __name__ == '__main__':
    prepare_corpus()
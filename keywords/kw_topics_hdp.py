# from collections import defaultdict
# from gensim import corpora, models
# import json
# import sys
# import os
#
# # TOFIX!
#
# def extract_topics():
#     words = json.load(open('data/words.json', 'r'))
#
#     if len(sys.argv) < 2:
#         print('Specify the clustering result to process')
#         exit()
#     clustering_file = sys.argv[1]
#
#     clustering = json.load(open(clustering_file, 'r'))
#
#     reverse_clustering = defaultdict(list)
#     for uuid, cluster in clustering.items():
#         reverse_clustering[cluster].append(uuid)
#
#     print('Number of clusters:', len(set(reverse_clustering.keys())))
#
#     with open(clustering_file[:-5] + '_topics_hdp', 'w', encoding='utf-8', errors='replace') as out_file:
#         for cluster in sorted(reverse_clustering):
#             out_file.write('{}\t{}\n'.format('Cluster', cluster))
#
#             uuids = reverse_clustering[cluster]
#             cur_corpus = prepare_corpus(uuids)
#
#             tfidf = models.TfidfModel(cur_corpus)
#             tfidf_corpus = tfidf[cur_corpus]
#
#             inv_words = {v: k for k, v in words.items()}
#             dictionary = corpora.Dictionary.from_corpus(tfidf_corpus, id2word=inv_words)
#
#             hdp = models.HdpModel(tfidf_corpus, id2word=dictionary)
#
#             for topic in hdp.print_topics():
#                 out_file.write(topic[1] + '\n')
#             out_file.write('\n')
#
#
# class MyCorpus(object):
#     def __init__(self, uuids=None):
#         if uuids:
#             self.uuids = uuids
#         else:
#             self.uuids = None
#
#     def __iter__(self):
#
#         if not self.uuids:
#             self.uuids = sorted(os.listdir(dir_malwords))
#         else:
#             self.uuids = sorted([uuid + '_ss.txt' for uuid in self.uuids])
#
#         for bow_file in self.uuids:
#
#             word_vec = []
#
#             for line in open(os.path.join(dir_malwords, bow_file), 'rb'):
#                 word, count = line.split()
#                 word = word.decode('utf-8')
#                 count = int(count)
#
#                 if word not in words:
#                     continue
#
#                 word_id = words[word]
#
#                 word_vec.append((word_id, count))
#
#             yield word_vec
#
#
# def prepare_corpus(dir_malwords, words, dictionary, uuids=None):
#     """
#     Generates a gensim corpus object from the bag of words.
#
#     :param uuids: List of uuids
#     :return:
#     """
#
#     config = json.load(open('config.json'))
#     dir_malwords = config['dir_mini']
#     words = json.load(open('data/words.json', 'r'))
#
#     corpus_dir = 'data_gensim'
#
#     if not os.path.exists(corpus_dir):
#         os.mkdir(corpus_dir)
#
#     corpus = MyCorpus(uuids)
#
#     corpora.MmCorpus.serialize(os.path.join(corpus_dir, 'corpus.mm'), corpus)
#
#     new_corpus = corpora.MmCorpus(os.path.join(corpus_dir, 'corpus.mm'))
#     inv_words = {v: k for k, v in words.items()}
#     dictionary = corpora.Dictionary.from_corpus(new_corpus, id2word=inv_words)
#
#     print(type(new_corpus))
#     print(type(dictionary))
#
#     return new_corpus

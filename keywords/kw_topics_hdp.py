from collections import defaultdict
from gensim import corpora, models
from wf_gensim import pp_gensim
import json
import sys


def extract_topics():
    words = json.load(open('data/words.json', 'r'))

    if len(sys.argv) < 2:
        print('Specify the clustering result to process')
        exit()
    clustering_file = sys.argv[1]

    clustering = json.load(open(clustering_file, 'r'))

    reverse_clustering = defaultdict(list)
    for uuid, cluster in clustering.items():
        reverse_clustering[cluster].append(uuid)

    print('Number of clusters:', len(set(reverse_clustering.keys())))

    with open(clustering_file[:-5] + '_topics_hdp', 'w', encoding='utf-8', errors='replace') as out_file:
        for cluster in sorted(reverse_clustering):
            out_file.write('{}\t{}\n'.format('Cluster', cluster))

            uuids = reverse_clustering[cluster]
            cur_corpus = pp_gensim.prepare_corpus(uuids)

            tfidf = models.TfidfModel(cur_corpus)
            tfidf_corpus = tfidf[cur_corpus]

            inv_words = {v: k for k, v in words.items()}
            dictionary = corpora.Dictionary.from_corpus(tfidf_corpus, id2word=inv_words)

            hdp = models.HdpModel(tfidf_corpus, id2word=dictionary)

            for topic in hdp.print_topics():
                out_file.write(topic[1] + '\n')
            out_file.write('\n')


if __name__ == '__main__':
    extract_topics()

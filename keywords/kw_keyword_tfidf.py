from collections import Counter, defaultdict
from utilities import constants
import json
import os


def extract_keywords(config, result_file, n_kws=10):
    """
    Extracts the n_kws most relevant words out of each cluster.
    The most relevant words are the one which have the highest sum of tfidfs values.

    :param config: global configuration dictionary
    :param result_file: file containing the results of clustering
    :param n_kws: number of keywords to extract
    :return: dictionary mapping the n_kws most relevant words and their value
    """

    dir_store = config['dir_store']
    clustering = json.load(open(result_file, 'r'))

    reverse_clustering = defaultdict(list)
    keywds = []

    for uuid, cluster in clustering.items():
        reverse_clustering[cluster].append(uuid)

    print('Number of clusters:', len(set(reverse_clustering.keys())))

    out_path = os.path.join(
        constants.dir_d,
        constants.dir_kwd,
        os.path.split(result_file[:-5])[1] + '_keywords_tfidf'
    )

    with open(out_path, 'w', encoding='utf-8', errors='replace') as out_file:
        for cluster in sorted(reverse_clustering):
            out_file.write('{}\t{}\n'.format('Cluster', cluster))
            highest_tfidf = Counter()

            uuids = reverse_clustering[cluster]
            for uuid in uuids:
                tfidfs = json.load(open(os.path.join(dir_store, uuid), 'r'))
                for word in tfidfs:
                    highest_tfidf[word] += tfidfs[word]

            keywds.append(highest_tfidf.most_common(n_kws))

            for keyword, tfidf in highest_tfidf.most_common(n_kws):
                out_file.write('{}\t{}\n'.format(keyword, tfidf))
            out_file.write('\n')

    return keywds

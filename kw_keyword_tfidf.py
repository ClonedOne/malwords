from collections import Counter, defaultdict
import json
import sys
import os


def extract_keywords():
    config = json.load(open('config.json'))
    dir_malwords = config['dir_mini']
    dir_store = config['dir_store']
    core_num = config['core_num']
    dir_base = config['dir_base']

    if len(sys.argv) < 2:
        print('Specify the clustering result to process')
        exit()
    clustering_file = sys.argv[1]

    clustering = json.load(open(clustering_file, 'r'))

    reverse_clustering = defaultdict(list)
    for uuid, cluster in clustering.items():
        reverse_clustering[cluster].append(uuid)

    print('Number of clusters:', len(set(reverse_clustering.keys())))

    with open(clustering_file[:-5] + '_keywords_tfidf', 'w', encoding='utf-8', errors='replace') as out_file:
        for cluster in sorted(reverse_clustering):
            out_file.write('{}\t{}\n'.format('Cluster', cluster))
            highest_tfidf = Counter()

            uuids = reverse_clustering[cluster]
            for uuid in uuids:
                tfidfs = json.load(open(os.path.join(dir_store, uuid), 'r'))
                for word in tfidfs:
                    highest_tfidf[word] += tfidfs[word]

            for keyword, tfidf in highest_tfidf.most_common(10):
                out_file.write('{}\t{}\n'.format(keyword, tfidf))
            out_file.write('\n')


if __name__ == '__main__':
    extract_keywords()
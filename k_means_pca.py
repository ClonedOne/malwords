from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import json
import os

dir_store = ''
num_clusters_max = 51
num_clusters = 5
core_num = 4
max_iter = 1000


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    uuids = sorted(os.listdir(dir_store))

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    base_labels = get_base_labels(uuids)
    base_labels = np.asarray(base_labels)
    print('Base labels')
    print(base_labels)

    k_means = KMeans(n_clusters=num_clusters, n_jobs=core_num, max_iter=max_iter)
    computed_labels = k_means.fit_predict(data)

    # Evaluate clustering
    print('Clustering evaluation:', num_clusters)
    print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
    print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
    print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
    print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
    print('Completeness:', metrics.completeness_score(base_labels, computed_labels))
    print('Silhouette', metrics.silhouette_score(data, computed_labels, metric='euclidean'))
    print('-'*80)

    result_to_visualize(uuids, base_labels, computed_labels)


def get_base_labels(uuids):
    """
    Returns the ordered list of base labels from AVClass output

    :return: ordered list of labels
    """

    base_labels = []
    uuid_label = json.load(open('data/labels.json'))
    families = {'mydoom': 0,
                'neobar': 1,
                'gepys': 2,
                'lamer': 3,
                'neshta': 4,
                'bladabindi': 5
                }

    for uuid in uuids:
        base_labels.append(families[uuid_label[uuid]])

    return base_labels


#
# Visualization
#

def result_to_visualize(uuids, base_labels, computed_labels):
    """
    Generate a json file structured so it can be used for visualization 

    :param uuids: list of uuids
    :param base_labels: base truth labels
    :param computed_labels: clustering results 
    :return: 
    """

    out_dict = {'name': 'clustering', 'children': []}
    colors = {0: 'blue',
              1: 'yellow',
              2: 'red',
              3: 'green',
              4: 'orange',
              5: 'brown'
              }

    for i in range(num_clusters):
        child_dict = {'name': str(i), 'children': []}

        for j in range(len(computed_labels)):
            label = int(computed_labels[j])
            if label == i:
                true_label = int(base_labels[j])
                child_inner = {'name': uuids[j], 'color': colors[true_label]}
                child_dict['children'].append(child_inner)

        out_dict['children'].append(child_dict)

    graph_path = 'visualize/graph1.json'
    json.dump(out_dict, open(graph_path, 'w'), indent=2)


def test_kmeans_clusters(data, base_labels):

    for cur_num_clusters in range(2, num_clusters_max):
        k_means = KMeans(n_clusters=cur_num_clusters, n_jobs=core_num, max_iter=max_iter)
        computed_labels = k_means.fit_predict(data)

        # Evaluate clustering
        print('Clustering evaluation:', cur_num_clusters)
        print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
        print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
        print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
        print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
        print('Completeness:', metrics.completeness_score(base_labels, computed_labels))
        print('Silhouette', metrics.silhouette_score(data, computed_labels, metric='euclidean'))
        print('-'*80)


if __name__ == '__main__':
    cluster()
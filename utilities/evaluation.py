from sklearn import metrics
import bcubed


def evaluate_clustering(base_labels, computed_labels, data=None, metric='euclidean'):
    """
    Print evaluation values for the clustering results
    
    :return: 
    """

    # Converts labels list to dictionaries for the BCubed library
    base_dict = {k: {v} for k, v in dict(enumerate(base_labels)).items()}
    computed_dict = {k: {v} for k, v in dict(enumerate(computed_labels)).items()}
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    print('-' * 80)

    ars = metrics.adjusted_rand_score(base_labels, computed_labels)
    ami = metrics.adjusted_mutual_info_score(base_labels, computed_labels)
    fm = metrics.fowlkes_mallows_score(base_labels, computed_labels)
    h = metrics.homogeneity_score(base_labels, computed_labels)
    c = metrics.completeness_score(base_labels, computed_labels)
    p = bcubed.precision(base_dict, computed_dict)
    r = bcubed.recall(base_dict, computed_dict)
    fs = bcubed.fscore(p, r)

    print('Clustering evaluation')
    print('Number of clusters', num_clusters)
    print('Number of distinct families', len(set(base_labels)))
    print('Adjusted Rand index:', ars)
    print('Adjusted Mutual Information:', ami)
    print('Fowlkes-Mallows:', fm)
    print('Homogeneity:', h)
    print('Completeness:', c)
    print('BCubed Precision:', p)
    print('BCubed Recall:', r)
    print('BCubed FScore:', fs)

    if data is not None:
        sh = metrics.silhouette_score(data, computed_labels, metric=metric)
        print('Silhouette', sh)
        ret = (ars, ami, fm, h, c, p, r, fs, sh)
    else:
        ret = (ars, ami, fm, h, c, p, r, fs)

    print('-' * 80)
    return ret

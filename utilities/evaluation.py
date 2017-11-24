from visualization import vis_classification
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn import metrics
import numpy as np
import bcubed


def evaluate_clustering(base_labels, computed_labels, data=None, metric='euclidean', silent=False):
    """
    Print evaluation metrics for the clustering results
    
    :param base_labels: labels from a reference clustering
    :param computed_labels: lables assigned by the clustering
    :param data: the data matrix or a list of uuids
    :param metric: metric to use for the silhouette method
    :param silent: flag, if true avoid printing
    :return:
    """

    # Converts labels list to dictionaries for the BCubed library
    base_dict = {k: {v} for k, v in dict(enumerate(base_labels)).items()}
    computed_dict = {k: {v} for k, v in dict(enumerate(computed_labels)).items()}
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    ars = metrics.adjusted_rand_score(base_labels, computed_labels)
    ami = metrics.adjusted_mutual_info_score(base_labels, computed_labels)
    fm = metrics.fowlkes_mallows_score(base_labels, computed_labels)
    h = metrics.homogeneity_score(base_labels, computed_labels)
    c = metrics.completeness_score(base_labels, computed_labels)
    p = bcubed.precision(base_dict, computed_dict)
    r = bcubed.recall(base_dict, computed_dict)
    fs = bcubed.fscore(p, r)

    if not silent:
        print('-' * 80)
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
        sh = metrics.silhouette_score(data, computed_labels, metric=metric, random_state=42)

        if not silent:
            print('Silhouette', sh)

        ret = (ars, ami, fm, h, c, p, r, fs, sh)

    else:
        ret = (ars, ami, fm, h, c, p, r, fs)

    return ret


def evaluate_classification(y_test, y_test_fam, y_predicted, dan_costs=None):
    """
    Show detailed information of the classification performance per single class

    :param y_test: test labels
    :param y_test_fam: test labels with mnemonic
    :param y_predicted: predicted labels
    :param dan_costs: list of dan epoch costs
    :return:
    """

    f1s = f1_score(y_test, y_predicted, average=None)
    avg_f1 = f1_score(y_test, y_predicted, average='micro')

    print('F1 scores on test set:')
    print(f1s)
    print('Average f1 score: {}'.format(avg_f1))

    classes = sorted(set(y_test))
    n_classes = len(classes)

    classes_dict = dict(zip(classes, range(n_classes)))

    class_fam = {}
    for i in range(len(y_test_fam)):
        class_fam[classes_dict[y_test[i]]] = y_test_fam[i]

    fam_score = {}
    for fam_num, fam in class_fam.items():
        fam_score[fam] = f1s[fam_num]

    for fam, score in sorted(fam_score.items()):
        print('{:20} {:20}'.format(fam, score))

    if dan_costs:
        vis_classification.plot_net_costs(dan_costs)

    vis_classification.plot_confusion_matrix(y_test, y_test_fam, y_predicted)

    return avg_f1, f1s


def cluster_metrics(reference, computed):
    """
    Compute the clustering precision, recall, quality, with respect to a reference clustering.
    (See: Scalable, Behavior-Based Malware Clustering)

    :param reference: reference clustering
    :param computed: computed clustering
    :return: precision, recall, quality
    """

    if len(reference) != len(computed):
        print('Number of points not equal')
        return None, None, None

    n_points = len(reference)

    r_clusters = sorted(set(reference))
    c_clusters = sorted(set(computed))

    precisions = []
    recalls = []

    # Transform both computed and reference label lists into inverted indices of clusters
    i_reference = defaultdict(set)
    i_computed = defaultdict(set)

    for i in range(n_points):
        i_reference[reference[i]].add(i)
        i_computed[computed[i]].add(i)

    for i in c_clusters:
        precisions.append(max([len(i_computed[i] & i_reference[j]) for j in r_clusters]))

    for i in r_clusters:
        recalls.append(max([len(i_computed[j] & i_reference[i]) for j in c_clusters]))

    precision = np.sum(np.array(precisions)) / n_points
    recall = np.sum(np.array(recalls)) / n_points
    quality = precision * recall

    return precision, recall, quality

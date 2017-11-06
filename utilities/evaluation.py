from visualization import vis_classification
from sklearn.metrics import f1_score
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

    print('F1 scores on test set:')
    print(f1s)
    print('Average f1 score: {}'.format(f1_score(y_test, y_predicted, average='micro')))

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


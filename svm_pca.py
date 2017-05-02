from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import json
import os

dir_store = ''
core_num = 4


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

    x_train, x_test, y_train, y_test = train_test_split(data, base_labels, test_size=0.2, random_state=42)
    print(len(x_train), len(x_test), len(y_train), len(y_test))

    svc = SVC(decision_function_shape='ovr')
    lin_svc = LinearSVC()

    svc.fit(x_train, y_train)
    lin_svc.fit(x_train, y_train)

    svc_score = svc.score(x_test, y_test)
    lin_svc_score = lin_svc.score(x_test, y_test)

    print('SVC score:', svc_score)
    print('Linear SVC score:', lin_svc_score)


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


if __name__ == '__main__':
    cluster()
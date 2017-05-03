from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from utilities import utils
import numpy as np
import json
import os

dir_store = ''
core_num = 1


def classify():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, core_num

    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = core_num['core_num']
    uuids = sorted(os.listdir(dir_store))

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquire base labels')
    base_labels = utils.get_base_labels(uuids)
    base_labels = np.asarray(base_labels)

    print('Split training and testing data')
    x_train, x_test, y_train, y_test = train_test_split(data, base_labels, test_size=0.2, random_state=42)
    print(len(x_train), len(x_test), len(y_train), len(y_test))

    print('Traing SVMs')
    svc = SVC(decision_function_shape='ovr')
    lin_svc = LinearSVC()

    svc.fit(x_train, y_train)
    lin_svc.fit(x_train, y_train)

    print('Evaluation')
    svc_score = svc.score(x_test, y_test)
    lin_svc_score = lin_svc.score(x_test, y_test)

    print('SVC score:', svc_score)
    print('Linear SVC score:', lin_svc_score)


if __name__ == '__main__':
    classify()
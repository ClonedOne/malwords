from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from utilities import utils
import numpy as np
import os


def classify(config, matrix_file):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :return: 
    """

    dir_store = config['dir_store']

    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Split training and testing data')
    x_train, x_test, y_train, y_test = train_test_split(data, base_labels, test_size=0.2)
    # print(x_train)
    # print(x_test)
    print(len(x_train), len(x_test), len(y_train), len(y_test))

    print('Traing SVMs')
    svc = SVC(kernel='linear')
    lin_svc = LinearSVC()

    svc.fit(x_train, y_train)
    lin_svc.fit(x_train, y_train)

    svc_pred = svc.predict(x_test)
    lin_svc_pred = lin_svc.predict(x_test)

    print('Base test labels')
    print(y_test)

    print('SVC prediction')
    print(svc_pred)

    print('LinearSVC prediction')
    print(lin_svc_pred)

    print('Evaluation')
    svc_score = svc.score(x_test, y_test)
    lin_svc_score = lin_svc.score(x_test, y_test)
    svc_prf = precision_recall_fscore_support(y_test, svc_pred, average='micro')
    lin_svc_prf = precision_recall_fscore_support(y_test, lin_svc_pred, average='micro')

    print('SVC score:', svc_score)
    print('Linear SVC score:', lin_svc_score)
    print('SVC Precision Recall Fscore: {} {} {}'.format(svc_prf[0], svc_prf[1], svc_prf[2]))
    print('Linear SVC Precision Recall Fscore: {} {} {}'.format(lin_svc_prf[0], lin_svc_prf[1], lin_svc_prf[2]))

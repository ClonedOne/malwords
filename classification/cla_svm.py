from sklearn.externals import joblib
from sklearn.metrics import f1_score
from utilities import constants
from sklearn.svm import SVC
import os


# noinspection PyUnusedLocal
def classify(xm_train, xm_dev, xm_test, y_train, y_dev, y_test, config):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :param xm_train: Training data matrix
    :param xm_dev: Development data matrix
    :param xm_test: Testing data matrix
    :param y_train: List of train set labels
    :param y_dev: List of dev set labels
    :param y_test: List of test set labels
    :param config: Global configuration dictionary
    :return: Predicted test labels and trained model
    """

    modifier = 'linear'

    svc = SVC(kernel=modifier)

    print('Training')
    svc.fit(xm_train, y_train)

    print('Prediction')
    train_predicted = svc.predict(xm_train)
    dev_predicted = svc.predict(xm_dev)

    train_score = f1_score(y_train, train_predicted, average='micro')
    dev_score = f1_score(y_dev, dev_predicted, average='micro')
    print('F1 score on train set: {}'.format(train_score))
    print('F1 score on dev set: {}'.format(dev_score))

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        '{}_{}_{}.pkl'.format('svc', modifier, xm_train.shape[0])
    )
    joblib.dump(svc, model_file)

    test_predicted = svc.predict(xm_test)
    print('F1 score on test set: {}'.format(f1_score(y_test, test_predicted, average='micro')))

    return test_predicted, svc, modifier

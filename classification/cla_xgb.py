from sklearn.externals import joblib
from sklearn.metrics import f1_score
from utilities import constants
import xgboost as xgb
import os


def classify(xm_train, xm_dev, xm_test, y_train, y_dev, y_test, config):
    """
    Classify the documents using a Random Forest Classifier and the AVClass labels as base truth.

    :param xm_train: Training data matrix
    :param xm_dev: Development data matrix
    :param xm_test: Testing data matrix
    :param y_train: List of train set labels
    :param y_dev: List of dev set labels
    :param y_test: List of test set labels
    :param config: Global configuration dictionary
    :return: Predicted test labels and trained model
    """

    modifier = 'multi'

    clas = xgb.XGBClassifier(
        seed=42,
        nthread=config['core_num'],
        silent=False,
    )

    print('Training')
    clas.fit(xm_train, y_train)

    print('Prediction')
    train_predicted = clas.predict(xm_train)
    dev_predicted = clas.predict(xm_dev)

    train_score = f1_score(y_train, train_predicted, average='micro')
    dev_score = f1_score(y_dev, dev_predicted, average='micro')
    print('F1 score on train set: {}'.format(train_score))
    print('F1 score on dev set: {}'.format(dev_score))

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        '{}_{}_{}.pkl'.format('xgb', modifier, xm_train.shape[0])
    )
    joblib.dump(clas, model_file)

    test_predicted = clas.predict(xm_test)
    print('F1 score on test set: {}'.format(f1_score(y_test, test_predicted, average='micro')))

    return test_predicted, clas, modifier

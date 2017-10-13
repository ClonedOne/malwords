from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from utilities import constants
from utilities import output
import os


def classify(config, train, test, x_test, y_train, y_test):
    """
    Classify the documents using a Random Forest Classifier and the AVClass labels as base truth.

    :param config: Global configuration dictionary
    :param train: Training data matrix
    :param test: Testing data matrix
    :param x_test: List of test set uuids
    :param y_train: List of train set labels
    :param y_test: List of test set labels
    :return: Classification label and trained model
    """

    criteria = ['gini', 'entropy']
    n_jobs = config['core_num']

    for criterion in criteria:
        randf = RandomForestClassifier(
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=42
        )

        print('Training')
        randf.fit(train, y_train)

        print('Prediction')
        classification_labels = randf.predict(test)

        test_score = f1_score(y_test, classification_labels, average='micro')
        print('F1 score of test: {}'.format(test_score))

    # output.out_classification(dict(zip(x_test, classification_labels.tolist())), 'linear', 'svm')

    # model_file = os.path.join(constants.dir_d, constants.dir_mod, 'svm_{}_{}.pkl'.format('linear', test.shape[0]))
    # joblib.dump(svc, model_file)

    return None, None

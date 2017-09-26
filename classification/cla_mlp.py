from sklearn.random_projection import SparseRandomProjection
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from utilities import constants
from utilities import output
import os


def classify(config, train, test, x_test, y_train, y_test):
    """
    Classify the documents using SVM and the AVClass labels as base truth.

    :param config: Global configuration dictionary
    :param train: Training data matrix
    :param test: Testing data matrix
    :param x_test: List of test set uuids
    :param y_train: List of train set labels
    :param y_test: List of test set labels
    :return: Classification label and trained model
    """

    core_num = config['core_num']
    components = 10000
    srp = SparseRandomProjection(n_components=components)

    if train.shape[1] > components:
        print('Dimensionality reduction through random projection')
        train = srp.fit_transform(train)

    print('10-fold cross validation')
    mlp = MLPClassifier(
        hidden_layer_sizes=(
            train.shape[1],
            int(train.shape[1] / 1.5),
            int(train.shape[1] / 2),
            int(train.shape[1] / 3),
            int(train.shape[1] / 4),
            len(set(y_train))
        ),
        max_iter=500,
        solver='adam',
        verbose=True
    )

    scores = cross_val_score(mlp, train, y_train, cv=10, n_jobs=core_num, scoring='f1_micro')

    print('F1 Scores of cross validation: {}'.format(scores))
    print('Average F1 score: {}'.format(sum(scores) / len(scores)))

    print('Training and prediction')
    mlp.fit(train, y_train)

    if test.shape[1] > components:
        print('Dimensionality reduction through random projection')
        test = srp.transform(test)

    classification_labels = mlp.predict(test)

    test_score = f1_score(y_test, classification_labels, average='micro')
    print('F1 score of test: {}'.format(test_score))

    output.out_classification(dict(zip(x_test, classification_labels.tolist())), 'gd', 'mlp')

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'mlp_{}_{}.pkl'.format(mlp.n_layers_, len(test)))
    joblib.dump(mlp, model_file)

    return classification_labels, mlp

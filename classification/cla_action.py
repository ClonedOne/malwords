from utilities import constants, interaction
from classification import cla_svm, cla_mlp
from helpers import loader_tfidf
import numpy as np


def classify(samples_data, config):
    """
    Perform a clustering or classification step.

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()
    y_train = samples_data.fam_num[samples_data['train'] == 1].tolist()
    y_test = samples_data.fam_num[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    cla = ''
    while cla == '':
        cla = input(constants.msg_cla)

        if cla == 'svm':
            train, test, sparse = select_data(config, uuids, x_train, x_test)
            return cla_svm.classify(config, train, test, x_test, y_train, y_test)

        elif cla == 'mlp':
            train, test, sparse = select_data(config, uuids, x_train, x_test)
            return cla_mlp.classify(config, train, test, x_test, y_train, y_test)

        elif cla == 's':
            return None, None

        elif cla == 'q':
            exit()

        else:
            print('Not a valid input\n')
            cla = ''


def select_data(config, uuids, x_train, x_test):
    """
    Asks the user for the data to operate upon.
     - work on the full vectors (sparse = true),
     - work on a reduced data matrix (sparse = false)

    :param config: Global configuration dictionary
    :param uuids: Lis of uuids for the selected dataset
    :param x_train: List of train set uuids
    :param x_test: List of test set uuids
    :return: train and test data matrices and the sparse flag
    """

    uuid_index = dict(zip(uuids, range(len(uuids))))

    sparse = interaction.ask_yes_no(constants.msg_sparse)

    if sparse:
        train = loader_tfidf.load_tfidf(config, x_train, dense=False, ordered=True)
        test = loader_tfidf.load_tfidf(config, x_test, dense=False, ordered=True)

    else:
        reduced = np.loadtxt(interaction.ask_file(constants.msg_data_red))

        train_pos = [uuid_index[uuid] for uuid in x_train]
        test_pos = [uuid_index[uuid] for uuid in x_test]

        train = np.take(reduced, train_pos, axis=0)
        test = np.take(reduced, test_pos, axis=0)

    return train, test, sparse

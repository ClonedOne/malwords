from classification import cla_svm, cla_rfc, cla_xgb, cla_dan
from utilities import constants, interaction, output, evaluation
from helpers import loader_tfidf
import numpy as np


def classify(samples_data, config):
    """
    Perform a clustering or classification step.

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    clas = {
        'svm': cla_svm,
        'dan': cla_dan,
        'rfc': cla_rfc,
        'xgb': cla_xgb
    }

    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_dev = samples_data.index[samples_data['dev'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()
    y_train = samples_data.fam_num[samples_data['train'] == 1].tolist()
    y_dev = samples_data.fam_num[samples_data['dev'] == 1].tolist()
    y_test = samples_data.fam_num[samples_data['test'] == 1].tolist()
    y_test_fam = samples_data.family[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    cla = interaction.ask_action(constants.msg_cla, set(clas.keys()))
    if cla == 's':
        return None, None, None

    xm_train, xm_dev, xm_test = select_data(config, x_train, x_dev, x_test)
    y_predicted, model, modifier = clas[cla].classify(
        xm_train,
        xm_dev,
        xm_test,
        y_train,
        y_dev,
        y_test,
        config,
        {}
    )

    output.out_classification(dict(zip(x_test, y_predicted.tolist())), modifier, cla)

    if cla == 'dan':
        evaluation.evaluate_classification(model[0], y_test_fam, y_predicted, model[1])
    else:
        evaluation.evaluate_classification(y_test, y_test_fam, y_predicted, None)


def select_data(config, x_train, x_dev, x_test):
    """
    Asks the user for the data to operate upon.
     - work on the full vectors (sparse = true),
     - work on a reduced data matrix (sparse = false)

    :param config: Global configuration dictionary
    :param x_train: List of train set uuids
    :param x_dev: List of dev set uuids
    :param x_test: List of test set uuids
    :return: data matrices
    """

    sparse = interaction.ask_yes_no(constants.msg_sparse)

    if sparse:
        xm_train = loader_tfidf.load_tfidf(config, x_train, dense=False, ordered=True)
        xm_dev = loader_tfidf.load_tfidf(config, x_dev, dense=False, ordered=True)
        xm_test = loader_tfidf.load_tfidf(config, x_test, dense=False, ordered=True)

    else:
        xm_train = np.loadtxt(interaction.ask_file(constants.msg_data_train))
        xm_dev = np.loadtxt(interaction.ask_file(constants.msg_data_dev))
        xm_test = np.loadtxt(interaction.ask_file(constants.msg_data_test))

    return xm_train, xm_dev, xm_test

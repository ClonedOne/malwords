from classification import cla_svm, cla_rand_forest, cla_xgb, cla_dan
from utilities import constants, interaction, output
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from helpers import loader_tfidf
import plotly.graph_objs as go
import plotly.offline as ply
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
        'rand': cla_rand_forest,
        'xgb': cla_xgb
    }

    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_dev = samples_data.index[samples_data['dev'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()
    y_train = samples_data.fam_num[samples_data['train'] == 1].tolist()
    y_dev = samples_data.fam_num[samples_data['dev'] == 1].tolist()
    y_test = samples_data.fam_num[samples_data['test'] == 1].tolist()
    y_test_fams = samples_data.family[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    cla = ''
    while cla == '':
        cla = input(constants.msg_cla)

        if cla in clas:
            xm_train, xm_dev, xm_test, sparse = select_data(config, x_train, x_dev, x_test)
            y_predicted, model, modifier = clas[cla].classify(xm_train, xm_dev, xm_test, y_train, y_dev, y_test, config)

            output.out_classification(dict(zip(x_test, y_predicted.tolist())), modifier, cla)

            show_detailed_score(y_test, y_test_fams, y_predicted)

            return y_predicted, model, modifier

        elif cla == 's':
            return None, None, None

        elif cla == 'q':
            exit()

        else:
            print('Not a valid input\n')
            cla = ''


def select_data(config, x_train, x_dev, x_test):
    """
    Asks the user for the data to operate upon.
     - work on the full vectors (sparse = true),
     - work on a reduced data matrix (sparse = false)

    :param config: Global configuration dictionary
    :param x_train: List of train set uuids
    :param x_dev: List of dev set uuids
    :param x_test: List of test set uuids
    :return: data matrices and the sparse flag
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

    return xm_train, xm_dev, xm_test, sparse


def show_detailed_score(y_test, y_test_fams, y_predicted):
    """
    Show detailed information of the classification performance per single class

    :param y_test: test labels
    :param y_test_fams: test labels with mnemonic
    :param y_predicted: predicted labels
    :return:
    """

    classes = sorted(set(y_test))
    n_classes = len(classes)

    classes_dict = dict(zip(classes, range(n_classes)))

    f1s = f1_score(y_test, y_predicted, average=None)

    print('F1 score on dev set:')
    print(f1s)
    print('Average f1 score: {}'.format(f1_score(y_test, y_predicted, average='micro')))

    class_fam = {}
    for i in range(len(y_test_fams)):
        class_fam[classes_dict[y_test[i]]] = y_test_fams[i]

    fam_score = {}
    for fam_num, fam in class_fam.items():
        fam_score[fam] = f1s[fam_num]

    for fam, score in sorted(fam_score.items()):
        print('{:20} {:20}'.format(fam, score))

    cm = confusion_matrix(y_test, y_predicted).astype(float)
    for vec in cm:
        vec /= np.sum(vec)

    families = [class_fam[i] for i in sorted(class_fam.keys())]

    trace = go.Heatmap(z=cm, x=families, y=families)
    ply.iplot([trace], filename='conf_matrix_28k')

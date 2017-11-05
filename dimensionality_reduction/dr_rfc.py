from utilities import constants, interaction
from sklearn.externals import joblib
from collections import defaultdict
from helpers import loader_tfidf
import numpy as np
import json
import os


def get_important_feats(feats_weights, max_feats, inv_words, n_uuid):
    """
    Returns a list of indices of the `max_feats` most important features given feature weights.

    :param feats_weights: list of feature weights
    :param max_feats: maximum number of desired features
    :param inv_words: dictionary of words with their feature positional id
    :param n_uuid: number of total uuids
    :return:
    """

    importance = defaultdict(list)
    selected_feats = []
    selected_words = {}
    n_feats = 0
    i = 0

    for imp in feats_weights:
        importance[imp].append(i)
        i += 1

    for imp in sorted(list(importance.keys()), reverse=True):
        imp_feats = importance[imp]
        to_add = len(imp_feats)

        if n_feats + to_add > max_feats:
            to_add = max_feats - n_feats

        selected_feats += sorted(imp_feats)[:to_add]

        for feat in sorted(imp_feats)[:to_add]:
            selected_words[inv_words[feat]] = imp

        n_feats += to_add

        if n_feats == max_feats:
            break

    feat_file = os.path.join(constants.dir_d, constants.dir_mod, 'components_rfc_{}_{}'.format(max_feats, n_uuid))
    json.dump(selected_words, open(feat_file, 'w'), indent=2)

    return selected_feats


def load_selected_feats(config, uuids, selected):
    """
    Select the specified features from the data vectors by loading the data set in mini batches

    :param config: application configuration dictionary
    :param uuids: list of uuids to load
    :param selected: list of indices of selected features
    :return:
    """

    load_batch_size = config['batch_size']
    new_data = []
    t = 0

    while t < len(uuids):
        batch = loader_tfidf.load_tfidf(config, uuids[t: t + load_batch_size], dense=True, ordered=True)
        batch = np.take(batch, selected, axis=1)
        new_data.append(batch)
        t += load_batch_size

    return np.concatenate(new_data)


def reduce(config, components, uuids=None, x_train=None, x_dev=None, x_test=None):
    """
    Use a trained random forest classifier to select a reduced number of features from the data set.

    :param config: configuration dictionary
    :param components: number of desired components
    :param uuids: list of selected uuids
    :param x_train: List of train set uuids
    :param x_dev: List of dev set uuids
    :param x_test: List of test set uuids
    :return:
    """

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    inv_words = {value: key for key, value in words.items()}

    print('Performing feature selection using Random Forest Classifiers')

    rfc_file = interaction.ask_file(constants.msg_data_rfc)
    rfc = joblib.load(rfc_file)

    if uuids:
        n_uuid = len(uuids)
    else:
        n_uuid = len(x_train)

    selected_feats = get_important_feats(rfc.feature_importances_, components, inv_words, n_uuid)

    if uuids:
        data = load_selected_feats(config, uuids, selected_feats)
        matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'rfc_{}_{}.txt'.format(components, len(uuids)))
        np.savetxt(open(matrix_file, 'wb'), data)

    else:
        t_train = load_selected_feats(config, x_train, selected_feats)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'rfc_{}_{}_tr.txt'.format(components, len(t_train))
        )
        np.savetxt(open(matrix_file, 'wb'), t_train)

        t_dev = load_selected_feats(config, x_dev, selected_feats)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'rfc_{}_{}_dv.txt'.format(components, len(t_dev))
        )
        np.savetxt(open(matrix_file, 'wb'), t_dev)

        t_test = load_selected_feats(config, x_test, selected_feats)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'rfc_{}_{}_te.txt'.format(components, len(t_test))
        )
        np.savetxt(open(matrix_file, 'wb'), t_test)

        data = (t_train, t_dev, t_test)

    return data, rfc

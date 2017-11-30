from clustering import clu_hdbscan, clu_kmeans, clu_spectral, clu_dbscan, clu_birch
from utilities import constants, interaction, output, evaluation
from helpers import loader_tfidf
import numpy as np


def cluster(samples_data, config):
    """
    Perform a clustering or classification step.

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    clus = {
        'kmeans': clu_kmeans,
        'spectral': clu_spectral,
        'dbscan': clu_dbscan,
        'hdbscan': clu_hdbscan,
        'birch': clu_birch
    }

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    numerical_labels = samples_data.fam_num[samples_data['selected'] == 1].tolist()

    # Prompts the user to select an action
    clu = interaction.ask_action(constants.msg_clu, set(clus.keys()))
    if clu == 's':
        return None, None

    data = select_data(config, uuids)

    clustering_labels, model, modifier, data, metric = clus[clu].cluster(
        data,
        numerical_labels,
        config,
        {}
    )

    evaluation.evaluate_clustering(numerical_labels, clustering_labels, data, metric)

    output.out_clustering(dict(zip(uuids, clustering_labels.tolist())), modifier, clu)

    output.result_to_visualize(uuids, numerical_labels, clustering_labels)


def select_data(config, uuids):
    """
    Asks the user for the data to operate upon.
     - work on the full vectors (sparse = true),
     - work on the full vectors with mini batches (sparse = false, mini = true)
     - work on a reduced data matrix (sparse = false, mini = false)

    :param config: Global configuration dictionary
    :param uuids: List of uuids
    :return: data matrices
    """

    sparse = interaction.ask_yes_no(constants.msg_sparse)

    if sparse:
        data = loader_tfidf.load_tfidf(config, uuids, dense=False, ordered=True)
    else:
        mini = interaction.ask_yes_no(constants.msg_mini)
        if mini:
            data = uuids
        else:
            data = np.loadtxt(interaction.ask_file(constants.msg_data_red))

    return data

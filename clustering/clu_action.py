from clustering import clu_hdbscan, clu_kmeans, clu_kmeans_minibatch, clu_spectral, clu_dbscan
from utilities import constants, interaction


def cluster(samples_data, config):
    """
    Perform a clustering or classification step.

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    labels_num = samples_data.fam_num[samples_data['selected'] == 1].tolist()

    # Prompts the user to select an action
    clu = ''
    while clu == '':
        clu = input(constants.msg_clu)

        if clu == 'kmeans':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                data_matrix = None
            else:
                data_matrix = interaction.ask_file(constants.msg_data_train)
            clusters = interaction.ask_number(constants.msg_clusters)
            return clu_kmeans.cluster(config, data_matrix, clusters, uuids, labels_num, sparse=sparse)

        elif clu == 'mini_kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            return clu_kmeans_minibatch.cluster(config, clusters, uuids, labels_num)

        elif clu == 'spectral':
            clusters = interaction.ask_number(constants.msg_clusters)
            return clu_spectral.cluster(config, clusters, uuids, labels_num)

        elif clu == 'dbscan':
            return clu_dbscan.cluster(config, uuids, labels_num)

        elif clu == 'hdbscan':
            distance = interaction.ask_metric()
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            return clu_hdbscan.cluster(config, distance, uuids, labels_num, sparse=sparse)

        elif clu == 's':
            return None, None

        elif clu == 'q':
            exit()

        else:
            print('Not a valid input\n')
            clu = ''

from clustering import clu_hdbscan, clu_kmeans, clu_kmeans_minibatch, clu_spectral, clu_dbscan
from utilities import constants, interaction


def cluster(uuids, base_labels, config):
    """
    Perform a clustering or classification step.

    :param uuids: list of uuids
    :param base_labels: list of malware families
    :param config: configuration dictionary
    :return:
    """

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
            clu_kmeans.cluster(config, data_matrix, clusters, uuids, base_labels, sparse=sparse)

        elif clu == 'mini_kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            clu_kmeans_minibatch.cluster(config, clusters, uuids, base_labels)

        elif clu == 'spectral':
            clusters = interaction.ask_number(constants.msg_clusters)
            clu_spectral.cluster(config, clusters, uuids, base_labels)

        elif clu == 'dbscan':
            clu_dbscan.cluster(config, uuids, base_labels)

        elif clu == 'hdbscan':
            distance = interaction.ask_metric()
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            clu_hdbscan.cluster(config, distance, uuids, base_labels, sparse=sparse)

        elif clu == 's':
            return

        elif clu == 'q':
            exit()

        else:
            print('Not a valid input\n')
            clu = ''

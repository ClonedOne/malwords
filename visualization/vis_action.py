from visualization import vis_classification, vis_cluster, vis_data
from utilities import interaction, constants, utils
import json


def visualize(config, uuids, subset_labels):
    """
    Perform visualization operations

    :param config: dictionary containg application configuration options
    :param uuids: list of selected uuids
    :param subset_labels: list of labels for the uuid subset
    :return:
    """

    if interaction.ask_yes_no(constants.msg_vis_features):
        vis_cluster.plot_av_features(config, uuids)

    if interaction.ask_yes_no(constants.msg_vis_dataset):
        data_matrix = interaction.ask_file(constants.msg_vis_base)
        vis_data.plot_data(data_matrix, subset_labels)

    uuid_index = dict(zip(uuids, range(len(uuids))))
    base_labels = utils.get_base_labels()

    if interaction.ask_yes_no(constants.msg_visualize_cla):
        data = json.load(open(interaction.ask_file(constants.msg_results_cla), 'r'))
        y_true = [base_labels[uuid] for uuid in sorted(list(data.keys()))]
        y_pred = [data[uuid] for uuid in sorted(list(data.keys()))]
        uuid_pos = [uuid_index[uuid] for uuid in sorted(list(data.keys()))]

        vis_classification.plot_confusion_matrix(y_true, y_pred)

        data_matrix = interaction.ask_file(constants.msg_vis_base)
        vis_classification.plot_classification(data_matrix, uuid_pos, y_pred, y_true)

    if interaction.ask_yes_no(constants.msg_visualize_clu):
        data = json.load(open(interaction.ask_file(constants.msg_results_clu), 'r'))
        y_pred = [data[uuid] for uuid in sorted(list(data.keys()))]
        uuid_pos = [uuid_index[uuid] for uuid in sorted(list(data.keys()))]

        vis_cluster.plot_cluster_features(config, data)

        data_matrix = interaction.ask_file(constants.msg_vis_base)
        vis_cluster.plot_clustering(data_matrix, uuid_pos, y_pred)

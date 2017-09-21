from visualization import vis_classification, vis_cluster, vis_data
from utilities import interaction, constants, utils
import json


def visualize(samples_data, config):
    """
    Perform visualization operations

    :param samples_data: DataFrame with samples information
    :param config: dictionary containg application configuration options
    :return:
    """

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    families = samples_data.family[samples_data['selected'] == 1].tolist()

    if interaction.ask_yes_no(constants.msg_vis_features):
        vis_cluster.plot_av_features(uuids, config)

    if interaction.ask_yes_no(constants.msg_vis_dataset):
        data_matrix = interaction.ask_file(constants.msg_vis_base)
        vis_data.plot_data(data_matrix, families)

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

        if interaction.ask_yes_no(constants.msg_visualize_feature_clu):
            vis_cluster.plot_cluster_features(config, data)

        data_matrix = interaction.ask_file(constants.msg_vis_base)
        vis_data.plot_data(data_matrix, y_pred)

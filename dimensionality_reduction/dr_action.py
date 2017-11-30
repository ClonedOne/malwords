from dimensionality_reduction import dr_pca, dr_tsne, dr_rfc, dr_irfc
from utilities import constants, interaction


def dimensionality_reduction(samples_data, config):
    """
    Perform a dimensionality reduction step (or skip).

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    drs = {
        'pca': dr_pca,
        'tsne': dr_tsne,
        'rfc': dr_rfc,
        'irfc': dr_irfc
    }

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_dev = samples_data.index[samples_data['dev'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    dr = interaction.ask_action(constants.msg_dr, set(drs.keys()))
    if dr == 's':
        return None, None

    components = interaction.ask_number(constants.msg_components)
    to_cla = interaction.ask_yes_no(constants.msg_cla_clu)

    if to_cla:
        data, model = drs[dr].reduce(config, components, None, x_train, x_dev, x_test)

    else:
        data, model = drs[dr].reduce(config, components, uuids, None, None, None)

    return data, model

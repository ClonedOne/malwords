from dimensionality_reduction import dr_pca, dr_tsne, dr_rfc
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
        'rfc': dr_rfc
    }

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_dev = samples_data.index[samples_data['dev'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    dr = ''
    while dr == '':
        dr = input(constants.msg_dr)

        if dr in drs:
            components = interaction.ask_number(constants.msg_components)
            to_cla = interaction.ask_yes_no(constants.msg_cla_clu)
            if to_cla:
                data, model = drs[dr].reduce(config, components, None, x_train, x_dev, x_test)
            else:
                data, model = drs[dr].reduce(config, components, uuids, None, None, None)

            return data, model

        elif dr == 's':
            return None, None

        elif dr == 'q':
            exit()

        else:
            print('Not a valid input\n')
            dr = ''

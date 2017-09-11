from dimensionality_reduction import dr_pca, dr_svd, dr_lda, dr_tsne
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
        'svd': dr_svd,
        'tsne': dr_tsne,
        'lda': dr_lda
    }

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()
    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    dr = ''
    while dr == '':
        dr = input(constants.msg_dr)

        if dr in drs:
            components = interaction.ask_number(constants.msg_components)
            drs[dr].reduce(config, uuids, None, components, 'all')
            drs[dr].reduce(config, x_train, x_test, components, 'train')

        elif dr == 's':
            return

        elif dr == 'q':
            exit()

        else:
            print('Not a valid input\n')
            dr = ''

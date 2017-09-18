from utilities import constants, interaction
from classification import cla_svm, cla_mlp


def classify(samples_data, config):
    """
    Perform a clustering or classification step.

    :param samples_data: DataFrame with samples information
    :param config: configuration dictionary
    :return:
    """

    x_train = samples_data.index[samples_data['train'] == 1].tolist()
    x_test = samples_data.index[samples_data['test'] == 1].tolist()
    y_train = samples_data.fam_num[samples_data['train'] == 1].tolist()
    y_test = samples_data.fam_num[samples_data['test'] == 1].tolist()

    # Prompts the user to select an action
    cla = ''
    while cla == '':
        cla = input(constants.msg_cla)

        if cla == 'svm':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                train, test = None, None
            else:
                train = interaction.ask_file(constants.msg_data_train)
                test = interaction.ask_file(constants.msg_data_test)
            return cla_svm.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif cla == 'mlp':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                train, test = None, None
            else:
                train = interaction.ask_file(constants.msg_data_train)
                test = interaction.ask_file(constants.msg_data_test)
            return cla_mlp.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif cla == 's':
            return None, None

        elif cla == 'q':
            exit()

        else:
            print('Not a valid input\n')
            cla = ''

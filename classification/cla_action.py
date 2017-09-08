from utilities import constants, interaction
from classification import cla_svm, cla_mlp


def classify(x_train, x_test, y_train, y_test, config):
    """
    Perform a clustering or classification step.

    :param x_train: list of train set uuids
    :param x_test: list of test set uuids
    :param y_train: list of train set labels
    :param y_test: list of test set labels
    :param config: configuration dictionary
    :return:
    """

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
            cla_svm.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif cla == 'mlp':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                train, test = None, None
            else:
                train = interaction.ask_file(constants.msg_data_train)
                test = interaction.ask_file(constants.msg_data_test)
            cla_mlp.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif cla == 's':
            return

        elif cla == 'q':
            exit()

        else:
            print('Not a valid input\n')
            cla = ''

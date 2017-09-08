from utilities import constants, interaction
from keywords import kw_keyword_tfidf


def keywords_extraction(config):
    """
    Perform keywords extraction from the clustered data

    :param config: configuration dictionary
    :return:
    """

    # Prompts the user to select an action
    kw = ''
    while kw == '':
        kw = input(constants.msg_kw)

        if kw == 'tfidf':
            result_file = interaction.ask_file(constants.msg_results_cluster)
            kw_keyword_tfidf.extract_keywords(config, result_file)

        elif kw == 's':
            return

        elif kw == 'q':
            exit()

        else:
            print('Not a valid input\n')
            kw = ''

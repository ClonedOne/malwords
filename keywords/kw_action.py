from utilities import constants, interaction
from keywords import kw_keyword_tfidf


def keywords_extraction(config):
    """
    Perform keywords extraction from the clustered data

    :param config: configuration dictionary
    :return:
    """

    kws = {
        'tfidf': kw_keyword_tfidf
    }

    # Prompts the user to select an action
    kw = interaction.ask_action(constants.msg_kw, set(kws.keys()))
    if kw == 's':
        return

    result_file = interaction.ask_file(constants.msg_results_cluster)
    kw.extract_keywords(config, result_file)

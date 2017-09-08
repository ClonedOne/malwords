from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf, pp_js, pp_word_probs
from sklearn.model_selection import train_test_split
from utilities import constants, utils, interaction
from collections import Counter
import os


def pre_process(config):
    """
    Perform pre-processing steps and returns a list of sample uuids and a list or related labels.

    :param config: configuration dictionary
    :return: tuple of lists, uuids and malware families
    """

    # Create results data directories if needed
    utils.create_dirs()

    # Create AVClass input data if needed
    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_labels)):
        pp_avclass.prepare_vt(config)
        print('Please run the AVClass tool and relaunch')
        exit()

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_labels)):
        pp_labels.get_labels(config)

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_words)):
        pp_idf.get_idf(config)

    if len(os.listdir(config['dir_store'])) == 0:
        if interaction.ask_yes_no(constants.msg_memhist):
            pp_word_probs.get_word_probabilities(config, 3)
        pp_tfidf.get_tf_idf(config)

    # Select the data subset to operate upon
    uuids = pp_subset.subset(config)

    # Retrieve base labels
    base_labels = utils.get_base_labels_uuids(uuids)

    print('\nAcquired {} samples belonging to {} different families'.format(len(uuids), len(set(base_labels))))

    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_js)) and interaction.ask_yes_no(
            constants.msg_js):
        pp_js.get_js(config, uuids)

    return uuids, base_labels


# Helper methods

def show_data(uuids, base_labels):
    """
    Helper method to show the train and test data sets.

    :param uuids: list of uuids composing the whole data set
    :param base_labels: list of related malware family labels
    :return:
    """

    index_label = utils.get_index_labels()

    x_train, x_test, y_train, y_test = train_test_split(uuids, base_labels, test_size=0.2, random_state=30)

    print('\n{} train samples belonging to {} malware families'.format(len(x_train), len(set(y_train))))
    distribution = Counter(y_train)
    for family in distribution.most_common():
        print('Malware family: {} - Number of samples: {}'.format(index_label[family[0]], family[1]))

    print('\n{} test samples belonging to {} malware families'.format(len(x_test), len(set(y_test))))
    distribution = Counter(y_test)
    for family in distribution.most_common():
        print('Malware family: {} - Number of samples: {}'.format(index_label[family[0]], family[1]))

    return x_train, x_test, y_train, y_test

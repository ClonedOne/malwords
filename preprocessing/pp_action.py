from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf, pp_js, pp_word_probs
from sklearn.model_selection import train_test_split
from utilities import constants, interaction
from collections import Counter
from pathlib import Path
import os


def pre_process(config):
    """
    Perform pre-processing steps and returns a list of sample uuids and a list or related labels.

    :param config: configuration dictionary
    :return: pandas DataFrame with:
     * index: sorted list of all the uuids
     * family: malware family associated with the uuid
     * fam_num: numerical index of the malware family
     * selected: list of bools stating if the uuid was selected by the user
     * train: list of bools showing uuids in train set
     * test: list of bools showing uuids in test set
    """

    # Create results data directories if needed
    create_dirs()

    # Create AVClass input data if needed
    check_avclass_files(config)

    # Create features files if needed
    check_features_files(config)

    # Select the data subset to operate upon
    samples_data = pp_subset.subset(config)

    if not os.path.isfile(os.path.join(constants.dir_d, constants.dir_mat, constants.file_js)) \
            and interaction.ask_yes_no(constants.msg_js):
        uuids = samples_data.index[samples_data['selected'] == 1].tolist()
        pp_js.get_js(config, uuids)

    return samples_data


# Helper methods

def create_dirs():
    """
    Helper method to create the data directory structure if needed.

    :return:
    """

    (Path(constants.dir_d) / Path(constants.dir_clu)).mkdir(parents=True, exist_ok=True)
    (Path(constants.dir_d) / Path(constants.dir_kw)).mkdir(parents=True, exist_ok=True)
    (Path(constants.dir_d) / Path(constants.dir_mat)).mkdir(parents=True, exist_ok=True)
    (Path(constants.dir_d) / Path(constants.dir_cla)).mkdir(parents=True, exist_ok=True)
    (Path(constants.dir_d) / Path(constants.dir_vis)).mkdir(parents=True, exist_ok=True)


def check_avclass_files(config):
    """
    Helper method to check for the existance of AVClass related files

    :param config: configuration dictionary
    :return:
    """

    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_labels)):
        pp_avclass.prepare_vt(config)
        print('Please run the AVClass tool and relaunch')
        exit()

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_labels)):
        pp_labels.get_labels(config)


def check_features_files(config):
    """
    Helper method to check the features files

    :param config: configuration dictionary
    :return:
    """

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_words)):
        pp_idf.get_idf(config)

    if len(os.listdir(config['dir_store'])) == 0:
        if interaction.ask_yes_no(constants.msg_memhist):
            pp_word_probs.get_word_probabilities(config, 3)
        pp_tfidf.get_tf_idf(config)


def split_show_data(samples_data):
    """
    Helper method to split and show the train and test data sets.

    :param samples_data: DataFrame with samples information
    :return:
    """

    uuids = samples_data.index[samples_data['selected'] == 1].tolist()

    labels = samples_data.loc[uuids, 'family'].values

    x_train, x_test, y_train, y_test = train_test_split(uuids, labels, test_size=0.2, random_state=42)

    print('\n{} train samples belonging to {} malware families'.format(len(x_train), len(set(y_train))))
    for family in Counter(y_train).most_common():
        print('Malware family: {:^20} Number of samples: {:^6}'.format(family[0], family[1]))

    print('\n{} test samples belonging to {} malware families'.format(len(x_test), len(set(y_test))))
    for family in Counter(y_test).most_common():
        print('Malware family: {:^20} Number of samples: {:^6}'.format(family[0], family[1]))

    for uuid in x_train:
        samples_data.set_value(uuid, 'train', 1)
    for uuid in x_test:
        samples_data.set_value(uuid, 'test', 1)

    print('\n')
    print(samples_data.describe(include='all'))

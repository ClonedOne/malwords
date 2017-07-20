from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf, pp_js, pp_word_probs
from clustering import ca_hdbscan, ca_kmeans, ca_kmeans_minibatch, ca_spectral, ca_dbscan
from dimensionality_reduction import dr_pca, dr_svd, dr_lda, dr_tsne
from sklearn.model_selection import train_test_split
from classification import ca_svm, ca_mlp
from keywords import kw_keyword_tfidf
from visualization import vis_plot
from utilities import interaction
from collections import Counter
from utilities import constants
from utilities import utils
import json
import os


def main():
    config = json.load(open('config.json', 'r'))

    uuids, base_labels = pre_process(config)

    x_train, x_test, y_train, y_test = show_data(uuids, base_labels)

    dimensionality_reduction(uuids, x_train, x_test, config)

    cluster_classify(uuids, x_train, x_test, y_train, y_test, base_labels, config)

    visualize(uuids, x_train, x_test, y_train, y_test, base_labels)

    keywords_extraction(config)


# Main lifecycle

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


def visualize(uuids, x_train, x_test, y_train, y_test, base_labels):
    """
    Perform visualization operations

    :param y_test:
    :param y_train:
    :param x_train:
    :param uuids:
    :param x_test:
    :param base_labels:
    :return:
    """

    if interaction.ask_yes_no(constants.msg_visualization):
        data_matrix = interaction.ask_file(constants.msg_data_visualize)
        vis_plot.plot_data(data_matrix, base_labels)

    if interaction.ask_yes_no(constants.msg_visualize_ca):
        data = json.load(open(interaction.ask_file(constants.msg_results_ca), 'r'))
        classification = [data[uuid] for uuid in x_test]
        data_matrix = interaction.ask_file(constants.msg_data_visualize)
        vis_plot.plot_classification(data_matrix, classification, y_test)


def cluster_classify(uuids, x_train, x_test, y_train, y_test, base_labels, config):
    """
    Perform a clustering or classification step.
    
    :param uuids: list of uuids
    :param x_train:
    :param x_test:
    :param y_train: list of train set labels
    :param y_test: list of test set labels
    :param base_labels: list of malware families
    :param config: configuration dictionary
    :return: 
    """

    # Prompts the user to select an action
    ca = ''
    while ca == '':
        ca = input(constants.msg_ca)

        if ca == 'kmeans':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                data_matrix = None
            else:
                data_matrix = interaction.ask_file(constants.msg_data_train)
            clusters = interaction.ask_number(constants.msg_clusters)
            ca_kmeans.cluster(config, data_matrix, clusters, uuids, base_labels, sparse=sparse)

        elif ca == 'mini_kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            ca_kmeans_minibatch.cluster(config, clusters, uuids, base_labels)

        elif ca == 'spectral':
            clusters = interaction.ask_number(constants.msg_clusters)
            ca_spectral.cluster(config, clusters, uuids, base_labels)

        elif ca == 'dbscan':
            ca_dbscan.cluster(config, uuids, base_labels)

        elif ca == 'hdbscan':
            distance = interaction.ask_metric()
            ca_hdbscan.cluster(config, distance, uuids, base_labels)

        elif ca == 'svm':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                train, test = None, None
            else:
                train = interaction.ask_file(constants.msg_data_train)
                test = interaction.ask_file(constants.msg_data_test)
            ca_svm.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif ca == 'mlp':
            sparse = interaction.ask_yes_no(constants.msg_sparse)
            if sparse:
                train, test = None, None
            else:
                train = interaction.ask_file(constants.msg_data_train)
                test = interaction.ask_file(constants.msg_data_test)
            ca_mlp.classify(config, train, test, x_train, x_test, y_train, y_test, sparse=sparse)

        elif ca == 's':
            return

        elif ca == 'q':
            exit()

        else:
            print('Not a valid input\n')
            ca = ''


def dimensionality_reduction(uuids, x_train, x_test, config):
    """
    Perform a dimensionality reduction step (or skip).

    :param uuids: list of uuids
    :param x_train: list of uuids for training
    :param x_test: list of uuids for testing
    :param config: configuration dictionary
    :return: 
    """

    drs = {
        'pca': dr_pca,
        'svd': dr_svd,
        'tsne': dr_tsne,
        'lda': dr_lda
    }

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


def pre_process(config):
    """
    Perform pre-processing steps and returns a list of sample uuids and a list or related labels.
    
    :param config: configuration dictionary
    :return: tuple of lists, uuids and malware families
    """

    # Create results data directories if needed
    if not os.path.exists(constants.dir_d):
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dc))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dg))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dm))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dt))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_ds))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dv))

    # Create AVClass input data if needed
    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_labels)):
        pp_avclass.prepare_vt(config)
        print('Please run the AVClass tool and relaunch')
        exit()

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_labels)):
        pp_labels.get_labels(config)

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_words)):
        pp_idf.get_idf(config)

    if interaction.ask_yes_no(constants.msg_memhist):
        pp_word_probs.get_word_probabilities(config, 3)

    if len(os.listdir(config['dir_store'])) != len(os.listdir(config['dir_malwords'])):
        pp_tfidf.get_tf_idf(config)

    # Select the data subset to operate upon
    uuids = pp_subset.subset(config)

    # Retrieve base labels
    base_labels = utils.get_base_labels_uuids(uuids)

    print('\nAcquired {} samples belonging to {} different families'.format(len(uuids), len(set(base_labels))))

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


if __name__ == '__main__':
    main()

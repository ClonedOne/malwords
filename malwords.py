from clustering import ca_hdbscan, ca_kmeans, ca_kmeans_minibatch, ca_spectral, ca_dbscan
from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf, pp_js
from dimensionality_reduction import dr_pca, dr_svd, dr_lda, dr_tsne
from sklearn.model_selection import train_test_split
from classification import ca_svm, ca_mlp
from distances import compare_distances
from utilities import interaction
from collections import Counter
from utilities import constants
from utilities import utils
import json
import sys
import os


def main():
    config = json.load(open('config.json', 'r'))

    uuids, base_labels = pre_process(config)

    x_train, x_test, y_train, y_test = show_data(uuids, base_labels)

    dimensionality_reduction(uuids, x_train, x_test, config)

    cluster_classify(uuids, y_train, y_test, base_labels, config)


# Main lifecycle

def cluster_classify(uuids, y_train, y_test, base_labels, config):
    """
    Perform a clustering or classification step.
    
    :param uuids: list of uuids
    :param y_train: list of train set labels
    :param y_test: list of test set labels
    :param base_labels: list of malware families
    :param config: configuration dictionary
    :return: 
    """

    # Prompts the user to select an action
    ca = ""
    while ca == "":
        ca = input(constants.msg_ca)

        if ca == 'kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            data_matrix = interaction.ask_file(constants.msg_data_train)
            ca_kmeans.cluster(config, data_matrix, clusters, uuids, base_labels)

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
            data_matrix = interaction.ask_file(constants.msg_data_train)
            data_matrix_test = interaction.ask_file(constants.msg_data_test)
            ca_svm.classify(config, data_matrix, data_matrix_test, uuids, y_train, y_test, sparse=False)

        elif ca == 'mlp':
            data_matrix = interaction.ask_file(constants.msg_data_train)
            data_matrix_test = interaction.ask_file(constants.msg_data_test)
            ca_mlp.classify(config, data_matrix, data_matrix_test, uuids, y_train, y_test, sparse=False)

        elif ca == 's':
            return

        elif ca == 'q':
            exit()

        else:
            print('Not a valid input\n')
            ca = ""


def dimensionality_reduction(uuids, x_train, x_test, config):
    """
    Perform a dimensionality reduction step (or skip).
    
    :param uuids: list of uuids
    :param x_train: list of uuids for training
    :param x_test: list of uuids for testing
    :param config: configuration dictionary
    :return: 
    """

    # Check if user has specified any action
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare-distance':
            compare_distances.compute_distances(config)

        else:
            print(constants.msg_argv)
            exit()

    # Prompts the user to select an action
    dr = ""
    while dr == "":
        dr = input(constants.msg_dr)

        if dr == 'pca':
            components = interaction.ask_number(constants.msg_components)
            dr_pca.get_pca(config, uuids, components, 'all')
            dr_pca.get_pca(config, x_train, components, 'train')
            dr_pca.get_pca(config, x_test, components, 'test')

        elif dr == 'svd':
            components = interaction.ask_number(constants.msg_components)
            dr_svd.get_svd(config, uuids, components, 'all')
            dr_svd.get_svd(config, x_train, components, 'train')
            dr_svd.get_svd(config, x_test, components, 'test')

        elif dr == 'tsne':
            components = interaction.ask_number(constants.msg_components)
            dr_tsne.get_tsne(config, uuids, components, 'all')

        elif dr == 'lda':
            components = interaction.ask_number(constants.msg_components)
            dr_lda.get_lda(config, uuids, components, 'all')
            dr_lda.get_lda(config, x_train, components, 'train')
            dr_lda.get_lda(config, x_test, components, 'test')

        elif dr == 's':
            return

        elif dr == 'q':
            exit()

        else:
            print('Not a valid input\n')
            dr = ""


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

    if len(os.listdir(config['dir_store'])) != len(os.listdir(config['dir_malwords'])):
        pp_tfidf.get_tf_idf(config)

    # Select the data subset to operate upon
    uuids = pp_subset.subset(config)

    # Retrieve base labels
    base_labels = utils.get_base_labels_uuids(uuids)

    print('\nAcquired {} samples belonging to {} different families'.format(len(uuids), len(set(base_labels))))

    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_js)):
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

    x_train, x_test, y_train, y_test = train_test_split(uuids, base_labels, test_size=0.2)

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

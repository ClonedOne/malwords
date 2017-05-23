from dimensionality_reduction import dr_pca, dr_svd, dr_lda, dr_kernel_pca, dr_tsne
from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf
from clustering import ca_hdbscan, ca_kmeans, ca_kmeans_minibatch
from distances import compare_distances
from classification import ca_svm
from utilities import interaction
from utilities import constants
import json
import sys
import os

# Action messages

msg_argv = 'Please select a valid action:\n' \
           'compare-distance --> show various distance metrics applies to the samples\n' \
           'q to quit\n'
msg_dr = 'Please select a dimensionality reduction technique:\n' \
         'pca\n' \
         'svd\n' \
         'kernel-pca\n' \
         'tnse\n' \
         'lda\n' \
         's to skip dimensionality reduction\n' \
         'q to quit\n'
msg_ca = 'Please select a clustering or classification technique:\n' \
         'kmeans for standard KMeans on feature selected data-set\n' \
         'mini_kmeans for mini batch KMeans\n' \
         'hdbscan for HDBSCAN on feature selected data-set\n' \
         'svm for linear SVM on feature selected data-set\n' \
         's to skip clustering/classification\n' \
         'q to quit\n'


def main():
    config = json.load(open('config.json', 'r'))

    pre_process(config)

    dimensionality_reduction(config)

    cluster_classify(config)


# Main lifecycle

def cluster_classify(config):
    """
    Perform a clustering or classification step.
    
    :param config: 
    :return: 
    """

    msg_clusters = 'clusters'
    msg_data = 'data matrix file'

    # Prompts the user to select an action
    ca = ""
    while ca == "":
        ca = input(msg_ca)

        if ca == 'kmeans':
            clusters = interaction.ask_number(msg_clusters)
            data_matrix = interaction.ask_file(msg_data)
            ca_kmeans.cluster(config, data_matrix, clusters)

        elif ca == 'mini_kmeans':
            clusters = interaction.ask_number(msg_clusters)
            ca_kmeans_minibatch.cluster(config, clusters)

        elif ca == 'hdbscan':
            data_matrix = interaction.ask_file(msg_data)
            distance = interaction.ask_metric()
            ca_hdbscan.cluster(config, data_matrix, distance)

        elif ca == 'svm':
            data_matrix = interaction.ask_file(msg_data)
            ca_svm.classify(config, data_matrix)

        elif ca == 's':
            return

        elif ca == 'q':
            exit()

        else:
            pre_process('Not a valid input\n')
            ca = ""


def dimensionality_reduction(config):
    """
    Perform a dimensionality reduction step (or skip).
    
    :param config: 
    :return: 
    """

    msg_components = 'components'

    # Check if user has specified any action
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare-distance':
            compare_distances.compute_distances(config)

        else:
            print(msg_argv)
            exit()

    # Prompts the user to select an action
    dr = ""
    while dr == "":
        dr = input(msg_dr)

        if dr == 'pca':
            components = interaction.ask_number(msg_components)
            dr_pca.get_pca(config, components)

        elif dr == 'svd':
            components = interaction.ask_number(msg_components)
            dr_svd.get_svd(config, components)

        elif dr == 'kernel-pca':
            components = interaction.ask_number(msg_components)
            dr_kernel_pca.get_kern_pca(config, components)

        elif dr == 'tnse':
            components = interaction.ask_number(msg_components)
            dr_tsne.get_tsne(config, components)

        elif dr == 'lda':
            components = interaction.ask_number(msg_components)
            dr_lda.get_lda(config, components)

        elif dr == 's':
            return

        elif dr == 'q':
            exit()

        else:
            pre_process('Not a valid input\n')
            dr = ""


def pre_process(config):
    """
    Perform pre-processing steps
    
    :param config: 
    :return: 
    """

    # Create results data directories if needed
    if not os.path.exists(constants.dir_d):
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dc))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dg))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dm))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dt))
        os.makedirs(os.path.join(constants.dir_d, constants.dir_dv))

    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_labels)):
        pp_avclass.prepare_vt(config)
        print('Please run the AVClass tool and relaunch')
        exit()

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_labels)):
        pp_labels.get_labels(config)

    if len(os.listdir(config['dir_mini'])) == 0:
        pp_subset.subset(config)
        print('Please unzip file (if necessary) and relaunch')
        exit()

    if not os.path.isfile(os.path.join(constants.dir_d, constants.json_words)):
        pp_idf.get_idf(config)

    if len(os.listdir(config['dir_store'])) == 0:
        pp_tfidf.get_tf_idf(config)


if __name__ == '__main__':
    main()

from clustering import ca_hdbscan, ca_kmeans, ca_kmeans_minibatch, ca_spectral, ca_dbscan
from dimensionality_reduction import dr_pca, dr_svd, dr_lda, dr_kernel_pca, dr_tsne
from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf, pp_js
from classification import ca_svm, ca_mlp
from distances import compare_distances
from utilities import interaction
from utilities import constants
import json
import sys
import os


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

    # Prompts the user to select an action
    ca = ""
    while ca == "":
        ca = input(constants.msg_ca)

        if ca == 'kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            data_matrix = interaction.ask_file(constants.msg_data)
            ca_kmeans.cluster(config, data_matrix, clusters)

        elif ca == 'mini_kmeans':
            clusters = interaction.ask_number(constants.msg_clusters)
            ca_kmeans_minibatch.cluster(config, clusters)

        elif ca == 'spectral':
            clusters = interaction.ask_number(constants.msg_clusters)
            ca_spectral.cluster(config, clusters)

        elif ca == 'dbscan':
            ca_dbscan.cluster(config)

        elif ca == 'hdbscan':
            distance = interaction.ask_metric()
            ca_hdbscan.cluster(config, distance)

        elif ca == 'svm':
            data_matrix = interaction.ask_file(constants.msg_data)
            ca_svm.classify(config, data_matrix, sparse=True)

        elif ca == 'mlp':
            data_matrix = interaction.ask_file(constants.msg_data)
            ca_mlp.classify(config, data_matrix, sparse=True)

        elif ca == 's':
            return

        elif ca == 'q':
            exit()

        else:
            print('Not a valid input\n')
            ca = ""


def dimensionality_reduction(config):
    """
    Perform a dimensionality reduction step (or skip).
    
    :param config: 
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
            dr_pca.get_pca(config, components)

        elif dr == 'svd':
            components = interaction.ask_number(constants.msg_components)
            dr_svd.get_svd(config, components)

        elif dr == 'kernel-pca':
            components = interaction.ask_number(constants.msg_components)
            dr_kernel_pca.get_kern_pca(config, components)

        elif dr == 'tnse':
            components = interaction.ask_number(constants.msg_components)
            dr_tsne.get_tsne(config, components)

        elif dr == 'lda':
            components = interaction.ask_number(constants.msg_components)
            dr_lda.get_lda(config, components)

        elif dr == 's':
            return

        elif dr == 'q':
            exit()

        else:
            print('Not a valid input\n')
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

    if not os.path.isfile(os.path.join(constants.dir_d, constants.file_js)):
        pp_js.get_js(config)


if __name__ == '__main__':
    main()

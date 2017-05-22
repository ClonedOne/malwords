from preprocessing import pp_avclass, pp_subset, pp_labels, pp_idf, pp_tfidf
from distances import compare_distances
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
         'q to quit\n'


def main():
    config = json.load(open('config.json', 'r'))

    pre_process(config)

    dimensionality_reduction(config)


# Main lifecycle

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
            print(msg_argv)
            exit()

    # Prompts the user to select an action
    action = ""
    while action == "":
        action = input(msg_dr)

        if action == 'q':
            exit()

        else:
            pre_process('Not a valid input\n')
            action = ""


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

from collections import defaultdict
from utilities import db_manager
from utilities import constants
import json
import os


def get_labels(config):
    """
    Generates json files mapping uuids to malware families.
    
    :param config: 
    :return: 
    """

    print('Reading labels from AVClass output')

    md5_uuid = db_manager.acquire_md5_uuid(config['dir_db'])
    remove_unavailable_uuids(config, md5_uuid)
    uuid_label, label_uuid = get_uuid_labels(md5_uuid)
    dump_labels(uuid_label, constants.json_labels)
    dump_labels(label_uuid, constants.json_inverted_labels)


def get_uuid_labels(md5_uuid):
    """
    Scans the label file produced by AVClass and generates the dictionary mapping uuids to labels
    
    :param md5_uuid: md5 to uuid mapping
    :return: dictionary mapping uuid to label
    """

    uuid_labels = {}
    inverted_labels = defaultdict(list)

    file_name = os.path.join(constants.dir_d, constants.file_labels)

    with open(file_name, 'r', encoding='utf-8', errors='replace') as labels_file:
        for line in labels_file:
            line = line.strip().split('\t')

            if 'SINGLETON' in line[1]:
                line[1] = line[1].split(':')[0]

            md5 = line[0]
            if md5 in md5_uuid:
                uuid = md5_uuid[md5]

                uuid_labels[uuid] = line[1]
                inverted_labels[line[1]].append(uuid)

    return uuid_labels, inverted_labels


def dump_labels(labels, out_file_name):
    """
    Outputs the labels dictionary to file.
    
    :param labels: labels dictionary
    :param out_file_name: name of the output file
    :return: 
    """

    out_path = os.path.join(constants.dir_d, out_file_name)

    with open(out_path, 'w', encoding='utf-8', errors='replace') as out_file:
        json.dump(labels, out_file, indent=2)


def remove_unavailable_uuids(config, md5_uuid):
    """
    Removes uuids which are not available in the dataset.

    :param config: configuration dictionary
    :param md5_uuid: dictionary to be modified
    :return:
    """

    to_remove = set()

    available_uuids = os.listdir(config['dir_malwords'])
    available_uuids = set([file_name.split('.')[0][:-3] for file_name in available_uuids])

    for md5, uuid in md5_uuid.items():
        if uuid not in available_uuids:
            to_remove.add(md5)

    for md5 in to_remove:
        md5_uuid.pop(md5, None)

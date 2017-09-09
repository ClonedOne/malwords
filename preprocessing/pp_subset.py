from utilities import interaction
from utilities import constants
from utilities import utils
import numpy as np
import pandas
import json
import os


def subset(config):
    """
    Subset the data-set for analysis.
    
    :param config: configuration dictionary
    :return: DataFrame with samples information
    """

    samples_data = create_dataframe(config)
    subset_type = ""

    while subset_type == "":
        subset_type = input(constants.msg_subset)

        if subset_type == 'l':
            get_labeled(samples_data)

        elif subset_type == 'k':
            load_samples(samples_data)

        elif subset_type == 'f':
            family = input(constants.msg_family)
            get_family(family, samples_data)

        elif subset_type == 's':
            load_small_set(samples_data)

        elif subset_type == 'j':
            json_file = interaction.ask_file(constants.msg_json)
            from_json(json_file, samples_data)

        elif subset_type == 'q':
            exit()

        else:
            subset_type = ""
            print(constants.msg_invalid)

    print(samples_data.describe(include='all'))
    return samples_data


def get_labeled(samples_data):
    """
    Take only samples for which there is a known malware family label.
    
    :param samples_data: DataFrame with samples information
    :return:
    """

    samples_data['selected'] = np.ones(len(samples_data.index))


def load_small_set(samples_data):
    """
    Load a small predefined subset of uuids.

    :param samples_data: DataFrame with samples information
    :return:
    """

    for uuid in constants.small_subset:
        if uuid in samples_data.index:
            samples_data.set_value(uuid, 'selected', 1)


def load_samples(samples_data):
    """
    Take samples of 7 specific families or a very small subset for testing. 
    
    :param samples_data: DataFrame with samples information
    :return:
    """

    familes = {'mydoom', 'gepys', 'lamer', 'neshta', 'bladabindi', 'flystudio', 'eorezo'}

    for uuid in samples_data.index:
        if samples_data.loc[uuid, 'family'] in familes:
            samples_data.set_value(uuid, 'selected', 1)


def from_json(json_file, samples_data):
    """
    Get samples specified in a json file.
    
    :param json_file: json file path
    :param samples_data: DataFrame with samples information
    :return: 
    """

    if not os.path.isfile(json_file):
        print('json file not found')
        exit()

    uuids = json.load(open(json_file))
    for uuid in uuids:
        if uuid in samples_data.index:
            samples_data.set_value(uuid, 'selected', 1)


def get_family(family_name, samples_data):
    """
    Get all samples of a specified family.

    :param samples_data: DataFrame with samples information
    :param family_name: chosen family name
    :return: 
    """

    if family_name not in samples_data.loc[:, 'family'].values:
        print('Could not find the chosen family')
        exit()

    for uuid in samples_data.index:
        if samples_data.loc[uuid, 'family'] == family_name:
            samples_data.set_value(uuid, 'selected', 1)


def create_dataframe(config):
    """
    Helper method to initialize a new DataFrame with the samples data with uuids as indices and the related families

    :param config: configuration dictionary
    :return: initialized DataFrame
    """

    uuid_label = json.load(open(os.path.join(constants.dir_d, constants.json_labels)))
    indices = set(utils.get_all_uuids(config['dir_malwords']))
    families = dict(zip(
        sorted(set(uuid_label.values()) - {'SINGLETON'}),
        range(len(set(uuid_label.values())) - 1)
    ))

    # Remove uuids whose malware family is unkown or unique
    to_remove = set()
    for uuid in indices:
        if uuid not in uuid_label or uuid_label[uuid] == 'SINGLETON':
            to_remove.add(uuid)
    indices = sorted(indices - to_remove)

    samples_data = pandas.DataFrame(index=indices, columns=constants.pd_columns)

    for uuid, label in uuid_label.items():
        if uuid in samples_data.index:
            samples_data.set_value(uuid, 'family', label)
            samples_data.set_value(uuid, 'fam_num', families[label])

    return samples_data

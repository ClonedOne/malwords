from collections import Counter
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
            load_labeled(samples_data)

        elif subset_type == 'k':
            load_samples(samples_data)

        elif subset_type == 'f':
            family = input(constants.msg_family)
            load_family(family, samples_data)

        elif subset_type == 's':
            load_balanced(samples_data, 100, 50)

        elif subset_type == 'b':
            load_balanced(samples_data, 100, 1000)

        elif subset_type == 'q':
            exit()

        else:
            subset_type = ""
            print(constants.msg_invalid)

    return samples_data


def load_balanced(samples_data, threshold_low, threshold_high):
    """
    Load a balanced subset of the data.
    Selects only those samples whose family appears more than a minim number of times.
    Selects only a maximum number of samples for each family.

    :param samples_data: DataFrame with samples information
    :param threshold_low: minium number of samples of the same family
    :param threshold_high: maximum number of samples per family
    :return:
    """

    temp = samples_data['family'].value_counts()
    families = set(temp[temp >= threshold_low].index)

    fam_count = Counter()

    for uuid in samples_data.index:
        cur_fam = samples_data.loc[uuid, 'family']
        if cur_fam in families and fam_count[cur_fam] < threshold_high:
            samples_data.at[uuid, 'selected'] = 1
            fam_count[cur_fam] += 1


def load_labeled(samples_data):
    """
    Take only samples for which there is a known malware family label.
    
    :param samples_data: DataFrame with samples information
    :return:
    """

    samples_data['selected'] = np.ones(len(samples_data.index))


def load_samples(samples_data):
    """
    Take samples of 7 specific families.
    
    :param samples_data: DataFrame with samples information
    :return:
    """

    families = {'mydoom', 'gepys', 'lamer', 'neshta', 'bladabindi', 'flystudio', 'eorezo'}

    for uuid in samples_data.index:
        if samples_data.loc[uuid, 'family'] in families:
            samples_data.at[uuid, 'selected'] = 1


def load_family(family_name, samples_data):
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
            samples_data.at[uuid, 'selected'] = 1


def create_dataframe(config):
    """
    Helper method to initialize a new DataFrame with the samples data with uuids as indices and the related families

    :param config: configuration dictionary
    :return: initialized DataFrame
    """

    uuid_label = json.load(open(os.path.join(constants.dir_d, constants.json_labels)))
    indices = set(utils.get_all_uuids(config['dir_store']))
    families = dict(zip(
        sorted(set(uuid_label.values()) - {'SINGLETON'}),
        range(len(set(uuid_label.values())) - 1)
    ))

    # Remove uuids whose malware family is unknown or unique
    to_remove = set()
    for uuid in indices:
        if uuid not in uuid_label or uuid_label[uuid] == 'SINGLETON':
            to_remove.add(uuid)
    indices = sorted(indices - to_remove)

    samples_data = pandas.DataFrame(index=indices, columns=constants.pd_columns)
    for col in constants.pd_columns[2:]:
        samples_data[col] = samples_data[col].astype(float)

    for uuid, label in uuid_label.items():
        if uuid in samples_data.index:
            samples_data.at[uuid, 'family'] = label
            samples_data.at[uuid, 'fam_num'] = families[label]

    return samples_data

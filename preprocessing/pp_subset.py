from utilities import interaction
from utilities import constants
from utilities import utils
from shutil import copyfile
import pandas
import json
import os


def subset(config):
    """
    Subset the data-set for analysis.
    
    :param config: configuration dictionary
    :return: 
    """

    samples_data = create_dataframe(config)
    print(samples_data.describe())

    uuids = utils.get_all_uuids(config['dir_malwords'])
    selected_subset = None
    subset_type = ""

    while subset_type == "":
        subset_type = input(constants.msg_subset)

        if subset_type == 'l':
            selected_subset = get_labeled(config)

        elif subset_type == 'k':
            selected_subset = load_samples(config)

        elif subset_type == 'f':
            family = input(constants.msg_family)
            selected_subset = get_family(family)

        elif subset_type == 's':
            selected_subset = load_samples(config, small=True)

        elif subset_type == 'j':
            json_file = interaction.ask_file(constants.msg_json)

            if not os.path.isfile(json_file):
                print('json file not found')
                exit()

            selected_subset = from_json(config, json_file)

        elif subset_type == 'q':
            exit()

        else:
            subset_type = ""
            print(constants.msg_invalid)

    return uuids, selected_subset


def get_labeled(config):
    """
    Take only samples for which there is a known malware family label.
    
    :param config: 
    :return: 
    """

    labels = json.load(open('data/labels.json'))
    not_labeled = []

    for file_name in sorted(os.listdir(config['dir_malwords'])):
        uuid = file_name.split('.')[0][:-3]

        if uuid not in labels:
            not_labeled.append(file_name)

    result = []
    for file_name in sorted(os.listdir(config['dir_malwords'])):
        if file_name not in not_labeled:
            result.append(file_name.split('.')[0][:-3])

    return sorted(result)


def load_samples(config, small=False):
    """
    Take samples of 7 specific families or a very small subset for testing. 
    
    :param config: 
    :param small: 
    :return: 
    """

    inv_labels = json.load(open('data/inverted_labels.json'))

    small_subset = [
        "761dd266-466c-41e3-8fab-a550adbe1a7c",
        "22b4014f-422c-4447-bfd1-a925bf33181e",
        "1c410f27-6b28-4ead-b2d1-53fcf3132394",
        "1dc440ca-0f47-4daf-a45c-5c9c7111da31",
        "859e3387-597c-4d0f-a539-7b74c5982a1c",
        "b790995f-8429-4b2b-96ef-f94bf000c1e1",
        "4905aa5a-9062-4d1d-9c72-96f1bd80bf3f",
        "ecc1e3df-bdf2-43c6-962e-ad2bc2de971a"
    ]

    if small:
        datasets = [small_subset, ]
    else:
        familes = ['mydoom', 'gepys', 'lamer', 'neshta', 'bladabindi', 'flystudio', 'eorezo']
        datasets = [inv_labels[family] for family in familes]

    available = set(os.listdir(config['dir_store']))
    to_return = []

    for dataset in datasets:
        for uuid in dataset:
            if uuid in available:
                to_return.append(uuid)

    return sorted(to_return)


def from_json(config, file_name):
    """
    Get samples specified in a json file.
    
    :param config: 
    :param file_name: 
    :return: 
    """

    result = []
    uuids = json.load(open(file_name))
    for uuid in sorted(os.listdir(config['dir_store'])):
        if uuid in uuids:
            result.append(uuid)

    return sorted(result)


def get_family(family):
    """
    Get all samples of a specified family.
    
    :param family:
    :return: 
    """

    inv_labels = json.load(open(os.path.join(constants.dir_d, constants.json_inverted_labels)))

    if family not in inv_labels:
        print('Malware family not found')
        exit()

    return sorted(inv_labels[family])


def copy_files(config, uuids):
    """
    Copy all the files in uuids.

    :param config:
    :param uuids:
    :return:
    """

    f_ext = '_ss.txt'

    for file_name in os.listdir(config['dir_malwords']):
        if '.gz' in file_name:
            f_ext = '_ss.txt.gz'

    for uuid in uuids:
        if os.path.isfile(os.path.join(config['dir_malwords'], uuid + f_ext)):
            copyfile(
                os.path.join(config['dir_malwords'], uuid + f_ext),
                os.path.join(config['dir_mini'], uuid + f_ext)
            )


def create_dataframe(config):
    """
    Helper method to initialize a new DataFrame with the samples data with uuids as indices and the related families

    :param config: configuration dictionary
    :return: initialized DataFrame
    """

    uuid_label = json.load(open(os.path.join(constants.dir_d, constants.json_labels)))
    indices = set(utils.get_all_uuids(config['dir_malwords']))
    families = dict(zip(
        sorted(set(uuid_label.values())),
        range(len(set(uuid_label.values())))
    ))


    # Remove uuids whose malware family is unkown
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

    print(sorted(set(samples_data['family'])))
    return samples_data

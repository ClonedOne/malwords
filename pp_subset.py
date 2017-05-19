from shutil import copyfile
import json
import sys
import os

f_ext = '_ss.txt.gz'
err_msg = 'Please choose the subset of data to extract:\n' \
          'l for labeled samples\n' \
          'k for 987 samples of families mydoom, neobar, gepys, lamer, neshta, bladabindi, flystudio\n' \
          's for 8 samples of families mydoom gepys, bladabindi, flystudio\n' \
          'j for json list of uuids'


def main():
    config = json.load(open('config.json'))

    if len(sys.argv) < 2:
        print(err_msg)
        exit()

    if sys.argv[1] == 'l':
        get_labeled(config)

    elif sys.argv[1] == 'k':
        load_samples(config)

    elif sys.argv[1] == 's':
        load_samples(config, small=True)

    elif sys.argv[1] == 'j':
        if len(sys.argv) < 3 or not os.path.isfile(sys.argv[2]):
            print(err_msg)
            exit()

        from_json(config, sys.argv[2])

    else:
        print(err_msg)


def get_labeled(config):
    labels = json.load(open('data/labels.json'))
    print('Total labeled:', len(labels))
    not_labeled = []

    for file_name in sorted(os.listdir(config['dir_malwords'])):
        uuid = file_name.split('.')[0][:-3]

        if uuid not in labels:
            not_labeled.append(file_name)

    print('Not labeled:', len(not_labeled))

    for file_name in sorted(os.listdir(config['dir_malwords'])):
        if file_name not in not_labeled:
            if os.path.isfile(os.path.join(config['dir_malwords'], file_name)):
                copyfile(
                    os.path.join(config['dir_malwords'], file_name),
                    os.path.join(config['dir_mini'], file_name)
                )


def load_samples(config, small=False):
    inv_labels = json.load(open('data/inverted_labels.json'))
    print('Number of malware families', len(inv_labels))

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
        familes = ['mydoom', 'neobar', 'gepys', 'lamer', 'neshta', 'bladabindi', 'flystudio']
        datasets = [inv_labels[family] for family in familes]

    for dataset in datasets:
        for f_name in dataset:
            if os.path.isfile(os.path.join(config['dir_malwords'], f_name + f_ext)):
                copyfile(
                    os.path.join(config['dir_malwords'], f_name + f_ext),
                    os.path.join(config['dir_mini'], f_name + f_ext)
                )


def from_json(config, file_name):
    uuids = json.load(open(file_name))
    for uuid in uuids:
        if os.path.isfile(os.path.join(config['dir_malwords'], uuid + f_ext)):
            copyfile(
                os.path.join(config['dir_malwords'], uuid + f_ext),
                os.path.join(config['dir_mini'], uuid + f_ext)
            )


if __name__ == '__main__':
    main()

from collections import defaultdict
from pprint import pprint
import db_manager


def main():
    db_path = '/home/yogaub/projects/projects_data/malrec/db'
    md5_uuid = db_manager.acquire_md5_uuid(db_path)
    label_uuid = get_inverted_labels(md5_uuid)
    print_inverted_labels(label_uuid)


def get_inverted_labels(md5_uuid):
    """
    Scans the label file produced by AVClass and generates the inverted dictionary of labels and md5s.
    
    :param md5_uuid: md5 to uuid mapping
    :return: 
    """

    inverted_labels = defaultdict(list)

    with open('labels.txt', 'r', encoding='utf-8', errors='replace') as labels_file:
        for line in labels_file:
            line = line.strip().split('\t')

            if 'SINGLETON' in line[1]:
                line[1] = line[1].split(':')[0]

            md5 = line[0]
            uuid = md5_uuid[md5]
            inverted_labels[line[1]].append(uuid)

    return inverted_labels


def print_inverted_labels(inverted_labels):
    """
    Outputs the inverted labels dictionary to file.
    
    :param inverted_labels: 
    :return: 
    """

    with open('inverted_labels.txt', 'w', encoding='utf-8', errors='replace') as inverted_file:
        for family, md5s in sorted(inverted_labels.items()):
            inverted_file.write(family + '\n')

            for md5 in md5s:
                inverted_file.write('\t' + md5 + '\n')

            inverted_file.write('\n')


if __name__ == '__main__':
    main()
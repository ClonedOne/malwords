from collections import defaultdict
from collections import Counter
from pprint import pprint
import json


def get_stats():
    stats_labels = get_labels_stats()
    print('Number of distinct samples:', stats_labels[0])
    print('Number of distinct families:', stats_labels[1])
    print('Number of singletons:', stats_labels[2])
    print('Average size of families:', stats_labels[3])
    pprint(stats_labels[4].most_common(10))
    threshold_high = 800
    threshold_low = 200
    uuids = inside_thresholds(stats_labels, threshold_low, threshold_high, stats_labels[5])
    json.dump(uuids, open('data/balanced.json', 'w'), indent=2)


def inside_thresholds(stats_labels, threshold_low, threshold_high, fam_uuids):
    """
    Poduce a list of uuids belonging to malware families whose population is inside the thresholds
    
    :return: 
    """

    print('Thresholds:', threshold_low, threshold_high)
    inside = []
    families = 0

    for fam, count in stats_labels[4].items():
        if fam == 'SINGLETON' or count < threshold_low:
            continue

        families += 1

        for i in fam_uuids[fam][:threshold_high]:
                inside.append(i)

    print('Inside thresholds:', families)
    print('Cumulative count:', len(inside))
    print('New average size:', len(inside)/families)

    return inside


def get_labels_stats():
    """
    Reads the labels file generated by AVClass and computes some statistics on it.
    
    :return: stats regarding AVClass labels
    """

    fam_counter = Counter()
    num_samples = 0
    num_singleton = 0
    fam_uuids = defaultdict(list)

    labels = json.load(open('data/labels.json'))
    for uuid, fam in labels.items():
        num_samples += 1

        if fam == 'SINGLETON':
            num_singleton += 1

        fam_counter[fam] += 1
        fam_uuids[fam].append(uuid)

    num_fam = len(fam_counter)

    avg_fam_size = 0.0
    for family in fam_counter.most_common():

        if family[0] == 'SINGLETON':
            continue

        avg_fam_size += family[1]

    avg_fam_size = avg_fam_size / num_fam

    return num_samples, num_fam, num_singleton, avg_fam_size, fam_counter, fam_uuids


if __name__ == '__main__':
    get_stats()

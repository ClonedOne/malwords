from utilities import constants
import subprocess
import time
import json
import os


def get_js(config, uuids):
    """
    Produces a distance matrix applying the Jensen-Shannon Distance to all the feature vectors.
    
    :param config: configuration dictionary
    :param uuids: list of uuids to operate on
    :return:
    """

    core_num = config['core_num']
    dir_malwords = config['dir_malwords']

    file_ext = '_ss.txt'
    zipped = 0
    if os.path.splitext(os.listdir(dir_malwords)[0])[1] == '.gz':
        file_ext = '_ss.txt.gz'
        zipped = 1

    uuids_paths = [os.path.join(dir_malwords, uuid + file_ext) for uuid in uuids]
    uuids_paths_file = os.path.join(constants.dir_d, 'uuids_paths.json')

    json.dump(uuids_paths, open(uuids_paths_file, 'w'), indent=2)
    time.sleep(1)

    proc = subprocess.Popen(['./js_dist', str(core_num), uuids_paths_file, str(zipped)])
    proc.wait()

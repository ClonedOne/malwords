from utilities import constants
import json
import os


def prepare_vt(config):
    """
    Prepares a formatted input file for the AVClass tool.
    
    :param config: 
    :return: 
    """

    print('Preparing AVClass input file.')

    reduced_vts = get_aggregated_vt(config['dir_vt'])
    create_avclass_input(reduced_vts)


def get_aggregated_vt(dir_vt):
    """
    Scans the VT reports in the folder and returns a list of reduced VT reports
    
    :param dir_vt: path to the VT reports
    :return: list of reduced VT reports
    """

    required_fields = ['md5', 'sha1', 'sha256', 'scan_date']
    reduced_vts = []

    for vt_report in sorted(os.listdir(dir_vt)):
        reduced_vt = {}

        with open(os.path.join(dir_vt, vt_report), 'r', encoding='utf-8', errors='replace') as vt_file:
            json_report = json.loads(vt_file.read())

            for field in required_fields:
                reduced_vt[field] = json_report[field]

            reduced_vt['av_labels'] = []

            for av, scan in json_report['scans'].items():
                if scan['detected']:
                    reduced_vt['av_labels'].append([av, scan['result']])

        reduced_vts.append(reduced_vt)

    return reduced_vts


def create_avclass_input(reduced_vts):
    """
    Creates the input file for AVClass labeling tool.
    
    :param reduced_vts: list of reduced VT reports
    :return: 
    """

    with open(os.path.join(constants.dir_d, constants.json_avc_input), 'w', encoding='utf-8', errors='replace') as avc:
        for reduced_vt in reduced_vts:
            avc.write(json.dumps(reduced_vt) + '\n')

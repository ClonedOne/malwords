from dimensionality_reduction import dr_action
from classification import cla_action
from visualization import vis_action
from preprocessing import pp_action
from clustering import clu_action
from keywords import kw_action
import json


def main():
    """
    Main application lifecycle. Loads configuration file and asks user which actions to perform.

    :return:
    """
    config = json.load(open('config.json', 'r'))

    samples_data = pp_action.pre_process(config)

    pp_action.split_show_data(samples_data)

    dr_action.dimensionality_reduction(samples_data, config)

    clu_action.cluster(samples_data, config)

    cla_action.classify(samples_data, config)

    vis_action.visualize(samples_data, config)

    kw_action.keywords_extraction(config)


if __name__ == '__main__':
    main()

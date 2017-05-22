import json
import os

dir_d = 'data'
dir_dc = 'data_cluster'
dir_dg = 'data_gensim'
dir_dv = 'data_visualize'


def main():
    config = json.load(open('config.json', 'r'))

    # Create results data directories if needed
    if not os.path.exists(dir_d):
        os.makedirs(os.path.join(dir_d, dir_dc))
        os.makedirs(os.path.join(dir_d, dir_dg))
        os.makedirs(os.path.join(dir_d, dir_dv))


if __name__ == '__main__':
    main()

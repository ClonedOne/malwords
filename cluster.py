import gzip
import os


dir_malwords = '/home/yogaub/projects/projects_data/malwords_results'


def main():
    for sample in sorted(os.listdir(dir_malwords)):

        with gzip.open(os.path.join(dir_malwords, sample), 'r') as words_file:

            for line in words_file:
                print(line)

        break


if __name__ == '__main__':
    main()
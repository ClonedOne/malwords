from collections import Counter
import sqlite3

"""
This module is used to obtain the name of the starting malware tested in each log file. 
Malware process names are the first 14 characters of the md5, the log file name is actually the uuid.
"""

db_name = 'panda.db'
table_name = 'samples'
column1 = 'uuid'
column2 = 'filename'
column3 = 'md5'


def acquire_malware_file_dict(dir_database_path):
    """
    Read the panda database file (SQLite) and returns a dictionary mapping panda log file names (uuids) to 
    malicious process names (md5 hashes) only the first 14 characters.

    :param dir_database_path:
    :return:
    """

    conn = sqlite3.connect(dir_database_path + '/' + db_name)
    c = conn.cursor()
    uuid_md5_dict = {}

    c.execute('SELECT {col1},{col2} FROM {tn}'.format(tn=table_name, col1=column1, col2=column3))
    all_rows = c.fetchall()
    for row in all_rows:
        uuid_md5_dict[row[0]] = row[1][:14]

    conn.close()
    return uuid_md5_dict


def acquire_malware_file_dict_full(dir_database_path):
    """
    Read the panda database file (SQLite) and returns a dictionary mapping panda log file names (uuids) to 
    malicious process names (md5 hashes).

    :param dir_database_path:
    :return:
    """

    conn = sqlite3.connect(dir_database_path + '/' + db_name)
    c = conn.cursor()
    uuid_md5_dict = {}

    c.execute('SELECT {col1},{col2} FROM {tn}'.format(tn=table_name, col1=column1, col2=column3))
    all_rows = c.fetchall()
    for row in all_rows:
        uuid_md5_dict[row[0]] = row[1]

    conn.close()
    return uuid_md5_dict


def acquire_md5_uuid(dir_database_path):
    """
    Read the panda database file (SQLite) and returns a dictionary mapping malicious process names (md5 hashes)
    to panda log file names (uuids).
    
    :param dir_database_path: 
    :return: 
    """

    conn = sqlite3.connect(dir_database_path + '/' + db_name)
    c = conn.cursor()
    md5_uuid_dict = {}
    md5s = Counter()

    c.execute('SELECT {col1},{col2} FROM {tn}'.format(tn=table_name, col1=column1, col2=column3))
    all_rows = c.fetchall()
    for row in all_rows:
        md5s[row[1]] += 1
        md5_uuid_dict[row[1]] = row[0]

    conn.close()

    for md5 in md5s:
        if md5s[md5] > 1:
            md5_uuid_dict.pop(md5, None)

    return md5_uuid_dict

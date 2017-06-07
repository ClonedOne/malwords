"""
This module is a collection of constants values used by the different modules.

"""

# Directories

dir_d = 'data'
dir_dc = 'data_cluster'
dir_dg = 'data_gensim'
dir_dm = 'data_matrix'
dir_dt = 'data_tfidf'
dir_dv = 'data_visualize'

# File names

file_labels = 'labels.txt'
file_js = 'jensen_shannon.txt'
json_avc_input = 'avc_input.json'
json_labels = 'labels.json'
json_inverted_labels = 'inverted_labels.json'
json_words = 'words.json'
json_dfs = 'dfs.json'
json_graph = 'graph1.json'

# Interface messages
msg_data = 'data matrix file'
msg_clusters = 'clusters'
msg_components = 'components'
msg_argv = '\nPlease select a valid action:\n' \
           'compare-distance --> show various distance metrics applies to the samples\n' \
           'q to quit\n'
msg_dr = '\nPlease select a dimensionality reduction technique:\n' \
         'pca\n' \
         'svd\n' \
         'tsne\n' \
         'lda\n' \
         's to skip dimensionality reduction\n' \
         'q to quit\n'
msg_ca = '\nPlease select a clustering or classification technique:\n' \
         'kmeans for standard KMeans on feature selected data-set\n' \
         'mini_kmeans for mini batch KMeans\n' \
         'hdbscan for HDBSCAN \n' \
         'svm for linear SVM \n' \
         'spectral for Spectral clustering using Jensen-Shannon metric\n' \
         'dbscan for DBSCAN clustering using Jensen-Shannon metric\n' \
         'mlp for multilayer Perceptron \n' \
         's to skip clustering/classification\n' \
         'q to quit\n'
msg_metric = 'Please select the desired metric\n' \
             'e for euclidean\n' \
             'c for cosine\n' \
             'j for jensen-shannon' \
             'q to quit\n'
msg_js = 'Do you want to compute the Jensen-Shannon distance matrix for the chose data? [y/n]\n ' \
         'It may take very long, deending on the number of selected samples\n'

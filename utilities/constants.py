"""
This module is a collection of constants values used by the different modules
"""

# Directories
dir_d = 'data'
dir_mod = 'model'
dir_mat = 'matrix'
dir_kwd = 'keyword'
dir_clu = 'clustering'
dir_vis = 'visualization'
dir_cla = 'classification'

# File names
file_js = 'jensen_shannon.txt'
json_dfs = 'dfs.json'
json_words = 'words.json'
json_graph = 'graph_{}.json'
file_labels = 'labels.txt'
json_labels = 'labels.json'
json_avc_input = 'avc_input.json'
json_words_probs = 'words_probs.json'
json_inverted_labels = 'inverted_labels.json'

# Interface messages
msg_json = 'json file'
msg_invalid = 'Not a valid input'
msg_data_dev = 'dev matrix'
msg_data_red = 'dimensionality reduced dataset'
msg_clusters = 'clusters'
msg_vis_base = 'baseline data matrix file'
msg_data_test = 'test matrix'
msg_data_train = 'train matrix'
msg_components = 'components'
msg_results_clu = 'clustering result file'
msg_results_cla = 'classification result file'
msg_results_cluster = 'clustering result file'
msg_data_visualize_base = 'base data matrix file to visualize'

msg_family = 'Please select the desired malware family\n'
msg_data_rfc = 'trained Random Forest Classifier model\n'

# Yes/No choices
msg_js = '\nWould you like to compute the Jensen-Shannon distance matrix for the chosen data? [y/n]\n'
msg_mini = '\nWould you like to use all the data with incremental mini-batches? (only K-Means!) [y/n]\n'
msg_sparse = '\nWould you like to use all the data as a sparse matrix? [y/n]\n'
msg_memhist = '\nWould you like to compute the tf-idf weights using memhist data? [y/n]\n'
msg_cla_clu = '\nAre you going to use the reduced data-set for classification? [y/n]\n'
msg_vis_dataset = '\nWould you like to visualize the dimensionality reduced data-set? [y/n]\n'
msg_vis_features = '\nWould you like to visualize the feature distributions in base data? [y/n]\n'
msg_visualize_cla = '\nWould you like to visualize the result of classification? [y/n]\n'
msg_visualize_clu = '\nWould you like to visualize the result of clustering? [y/n]\n'
msg_visualize_feature_clu = '\nWould you like to visualize the feature distribution of clustering? [y/n]\n'

#  Static lists
pd_columns = ['family', 'fam_num', 'selected', 'train', 'dev', 'test']

# Multiple choices
msg_subset = 'Please choose the subset of data to workon on:\n' \
             'l for all labeled samples\n' \
             'k for samples of families mydoom, gepys, lamer, neshta, bladabindi, flystudio, eorezo\n' \
             's for a small balanced subset\n' \
             'f for a single family\n' \
             'b for a balanced subset of samples\n' \
             'q to quit\n'

msg_dr = '\nPlease select a dimensionality reduction technique:\n' \
         'pca\n' \
         'svd\n' \
         'tsne\n' \
         'rfc\n' \
         's to skip dimensionality reduction\n' \
         'q to quit\n'

msg_cla = '\nPlease select a classification technique:\n' \
          'svm\n' \
          'dan\n' \
          'xgb\n' \
          'rand\n' \
          's to skip\n' \
          'q to quit\n'

msg_metric = '\nPlease select the desired metric\n' \
             'e for euclidean\n' \
             'c for cosine precomputed\n' \
             'c1 for cosine approximation\n' \
             'j for jensen-shannon precomputed\n' \
             'j1 for jensen-shannon\n' \
             'q to quit\n'

msg_clu = '\nPlease select a clustering technique:\n' \
          'kmeans\n' \
          'hdbscan\n' \
          'spectral\n' \
          'dbscan\n' \
          's to skip\n' \
          'q to quit\n'

msg_kw = '\nPlease select a keywords extraction technique:\n' \
         'tfidf\n' \
         's to skip\n' \
         'q to quit\n'

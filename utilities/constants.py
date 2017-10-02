"""
This module is a collection of constants values used by the different modules
"""

# Directories
dir_d = 'data'
dir_clu = 'd_clusterings'
dir_kw = 'd_keywords'
dir_mat = 'd_matrices'
dir_cla = 'd_classifications'
dir_vis = 'd_visualizations'
dir_mod = 'd_models'

# File names
file_labels = 'labels.txt'
file_js = 'jensen_shannon.txt'
json_avc_input = 'avc_input.json'
json_labels = 'labels.json'
json_inverted_labels = 'inverted_labels.json'
json_words = 'words.json'
json_words_probs = 'words_probs.json'
json_dfs = 'dfs.json'
json_graph = 'graph_{}.json'

# Interface messages
msg_data_red = 'dimensionality reduced dataset'
msg_vis_base = 'baseline data matrix file'
msg_data_visualize_base = 'base data matrix file to visualize'
msg_clusters = 'clusters'
msg_components = 'components'
msg_results_cla = 'classification result file'
msg_results_clu = 'clustering result file'
msg_results_cluster = 'clustering result file'
msg_json = 'json file'

msg_family = 'Please select the desired malware family\n'

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

msg_cla = '\nPlease select a classification technique:\n' \
          'svm for linear kernel SVM \n' \
          'mlp for multilayer Perceptron \n' \
          's to skip clustering/classification\n' \
          'q to quit\n'

msg_clu = '\nPlease select a clustering technique:\n' \
          'kmeans for standard KMeans on feature selected data-set\n' \
          'mini_kmeans for mini batch KMeans\n' \
          'hdbscan for HDBSCAN \n' \
          'spectral for Spectral clustering using Jensen-Shannon metric\n' \
          'dbscan for DBSCAN clustering using Jensen-Shannon metric\n' \
          's to skip clustering/classification\n' \
          'q to quit\n'

msg_metric = '\nPlease select the desired metric\n' \
             'e for euclidean\n' \
             'c for cosine\n' \
             'j for jensen-shannon' \
             'q to quit\n'

msg_js = '\nWould you like to compute the Jensen-Shannon distance matrix for the chosen data? [y/n]\n' \
         'It may take very long, depending on the number of selected samples\n'

msg_sparse = '\nWould you like to use all the data as a sparse matrix? [y/n]\n' \
             'It may take very long, depending on the number of selected samples\n'

msg_subset = 'Please choose the subset of data to workon on:\n' \
             'l for all labeled samples\n' \
             'k for samples of families mydoom, gepys, lamer, neshta, bladabindi, flystudio, eorezo\n' \
             's for 8 samples of families mydoom, gepys, bladabindi, flystudio\n' \
             'f for a single family\n' \
             'b for a balanced subset of samples\n' \
             'q to quit\n'

msg_vis_dataset = '\nWould you like to visualize the dimensionality reduced data-set? [y/n]\n'
msg_vis_features = '\nWould you like to visualize the feature distributions in base data? [y/n]\n'

msg_visualize_cla = '\nWould you like to visualize the result of classification? [y/n]\n'
msg_visualize_clu = '\nWould you like to visualize the result of clustering? [y/n]\n'
msg_visualize_feature_clu = '\nWould you like to visualize the feature distribution of clustering? [y/n]\n'

msg_memhist = '\nWould you like to compute the tf-idf weights using memhist data? [y/n]\n' \
              'It may take very long.\n'

msg_kw = '\nPlease select a keywords extraction technique:\n' \
         'tfidf for the words with highest weighted tf-idf\n' \
         's to skip keywords extraction\n' \
         'q to quit\n'

pd_columns = ['family', 'fam_num', 'selected', 'train', 'dev', 'test']

msg_invalid = 'Not a valid input'

small_subset = [
    "761dd266-466c-41e3-8fab-a550adbe1a7c",
    "22b4014f-422c-4447-bfd1-a925bf33181e",
    "1c410f27-6b28-4ead-b2d1-53fcf3132394",
    "1dc440ca-0f47-4daf-a45c-5c9c7111da31",
    "859e3387-597c-4d0f-a539-7b74c5982a1c",
    "b790995f-8429-4b2b-96ef-f94bf000c1e1",
    "4905aa5a-9062-4d1d-9c72-96f1bd80bf3f",
    "ecc1e3df-bdf2-43c6-962e-ad2bc2de971a"
]

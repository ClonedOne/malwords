#include <iostream>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>
#include <iterator>
#include <fstream>
#include <typeinfo>
#include <cstring>
#include <set>

#include <dirent.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "json.hpp"


using namespace Eigen;
using namespace std::chrono;
using json = nlohmann::json;


struct args_t_data {
    int core;
    std::vector<std::string> sub_vec;
    Eigen::SparseMatrix<double, Eigen::RowMajor> *data;
};

struct args_t_js {
    int core;
    std::vector<long> sub_vec;
};



//// Global variables

int NR_CPU;
int rows;
int cols;

json words;
json file_paths;

std::string command = "zcat ";

std::set<std::string> sorted_files;
Eigen::SparseMatrix<double, Eigen::RowMajor> vectors_matrix;
Eigen::MatrixXd js_distances;




//// Function declarations

void inline trim(std::string str);

std::string exec(const char *cmd);

template<typename Out>
void split(const std::string &s, char delim, Out result);

std::vector<std::string> split(const std::string &s, char delim);

inline double kl_divergence(const Eigen::ArrayXd &p, const Eigen::ArrayXd &q);

double js_divergence(const Eigen::ArrayXd &p, const Eigen::ArrayXd &q);

void *thread_distances(void *arg);

void *thread_acquire(void *arg);

Eigen::SparseMatrix<double, Eigen::RowMajor> acquire_data_multi();

void acquire_data(int cur_rows, std::vector<std::string> cur_files,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> *data_matrix);

Eigen::MatrixXd js_dist_multi();

long inline merge_int(int x, int y);

void inline split_long(long l, int *x, int *y);



//// MAIN

/**
 * Computes the Jensen-Shannon distance between all the Bag-of-Words files contained in a directory.
 * Outputs a .txt file with the distance matrix.
 * @return
 */
int main(int argc, char *argv[]) {

    // Get arguments from command line
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " NR_CPU PATH/TO/DIR" << std::endl;
        exit(1);
    }

    NR_CPU = std::stoi(argv[1]);
    std::ifstream i_p(argv[2]);
    i_p >> file_paths;
    for (std::string file_path : file_paths) sorted_files.insert(file_path);

    // Load valid word list with numerical index.
    std::ifstream i_w("words.json");
    i_w >> words;

    cols = words.size();
    rows = sorted_files.size();
    std::cout << "Number of files: " << sorted_files.size() << std::endl;

    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    vectors_matrix = acquire_data_multi();
    std::cout << "Number of non-zero elements: " << vectors_matrix.nonZeros() << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    duration<double, std::milli> time_span0 = t1 - t0;
    std::cout << "Data acquisition: " << time_span0.count() << " milliseconds." << std::endl;

    js_distances = Eigen::MatrixXd::Zero(rows, rows);
    js_dist_multi();

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double, std::milli> time_span1 = t2 - t1;
    std::cout << "JS computation: " << time_span1.count() << " milliseconds." << std::endl;

    std::ofstream js_file("jensen_shannon.txt");
    if (js_file.is_open())
        js_file << js_distances;
    js_file.close();

    return 0;
}




//// Data acquisition

/**
 * Scans through the directory and acquires data as vectors.
 */
void acquire_data(int cur_rows, std::vector<std::string> cur_files,
                  Eigen::SparseMatrix<double, Eigen::RowMajor> *data_matrix) {
    int row_iter = 0;

    for (std::string file_n : cur_files) {
        std::string result = exec((command + file_n).c_str());

        for (std::string line : split(result, '\n')) {
            if (line.empty()) continue;

            trim(line);
            std::vector<std::string> word_freq = split(line, ' ');

            if (words.find(word_freq[0]) != words.end())
                (*data_matrix).insert(row_iter, words[word_freq[0]]) = std::stoi(word_freq[1]);
        }

        row_iter++;

    }

    (*data_matrix).makeCompressed();
}


/**
 * Thread function for data acquisition.
 * @param arg
 * @return
 */
void *thread_acquire(void *arg) {
    args_t_data args = *(args_t_data *) arg;
    int cur_rows = args.sub_vec.size();
    acquire_data(cur_rows, args.sub_vec, args.data);
    std::cout << args.core << " - " << args.sub_vec.size() << " - " << (*args.data).nonZeros() << std::endl;
    return NULL;
}


/**
 * Acquire the data matrix from files, using multple threads.
 * @return
 */
Eigen::SparseMatrix<double, Eigen::RowMajor> acquire_data_multi() {
    Eigen::SparseMatrix<double, Eigen::RowMajor> data_matrix(rows, cols);
    pthread_t threads[NR_CPU];
    args_t_data thread_args[NR_CPU];

    int per_core = sorted_files.size() / NR_CPU;
    int extra = sorted_files.size() % NR_CPU;

    std::vector<std::string> vec_files(sorted_files.begin(), sorted_files.end());
    for (int i = 0; i < NR_CPU; i++) {

        //create a sub vector containing the files to operate on for each worker thread.
        int init = i * per_core;
        int end = i != NR_CPU - 1 ? (i + 1) * per_core : ((i + 1) * per_core) + extra;
        std::vector<std::string> sub_vec(&vec_files[init], &vec_files[end]);

        thread_args[i].core = i;
        thread_args[i].sub_vec = sub_vec;

        // Reserve space for each thread matrix
        int cur_rows = end - init;
        thread_args[i].data = new Eigen::SparseMatrix<double, Eigen::RowMajor>(cur_rows, cols);
        (*thread_args[i].data).reserve(cur_rows * 100000);

        pthread_create(&threads[i], NULL, thread_acquire, &thread_args[i]);
    }

    for (int i = 0; i < NR_CPU; ++i) {
        int init = i * per_core;
        int end = i != NR_CPU - 1 ? (i + 1) * per_core : ((i + 1) * per_core) + extra;

        pthread_join(threads[i], NULL);

        int cur_rows = end - init;
        data_matrix.middleRows(init, cur_rows) = *thread_args[i].data;
    }

    return data_matrix;
}






//// Jensen Shannon calculation

/**
 * Computes the Kullback–Leibler divergence of two dense vectors.
 * @param p
 * @param q
 * @return
 */
inline double kl_divergence(const Eigen::ArrayXd &p, const Eigen::ArrayXd &q) {
    return (p * Eigen::log(p / q)).sum();
}


/**
 * Computes the Jensen-Shannon divergence of two dense vectors.
 * @param p
 * @param q
 * @return
 */
double js_divergence(const Eigen::ArrayXd &p, const Eigen::ArrayXd &q) {

    // Apply Laplace Smoothing
    Eigen::ArrayXd p_prob = (p + 1) / (p.sum() + cols);
    Eigen::ArrayXd q_prob = (q + 1) / (q.sum() + cols);

    Eigen::ArrayXd m = (p_prob + q_prob) / 2;
    return sqrt((0.5 * kl_divergence(p_prob, m)) + (0.5 * kl_divergence(q_prob, m)));
}


/**
 * Thread function for JS distance computation.
 * @param data_matrix
 */
void *thread_distances(void *arg) {
    args_t_js args = *(args_t_js *) arg;
    Eigen::ArrayXd p, q;
    int i, j, old_i;
    old_i = -1;

    for (long comb : args.sub_vec) {
        split_long(comb, &i, &j);

        if (old_i != i) {
            p = Eigen::VectorXd(vectors_matrix.innerVector(i)).array();
            old_i = i;
        }

        q = Eigen::VectorXd(vectors_matrix.innerVector(j)).array();
        double div = js_divergence(p, q);

        js_distances(i, j) = div;
        js_distances(j, i) = div;
    }

    return NULL;
}


/**
 * Compute distances over multiple threads.
 * @return
 */
Eigen::MatrixXd js_dist_multi() {
    Eigen::MatrixXd js_dist;
    std::vector<long> combinations;
    combinations.reserve(rows * (rows - 1) / 2);

    for (int i = 0; i < rows; i++) {
        for (int j = i + 1; j < rows; j++) {
            combinations.push_back(merge_int(i, j));
        }
    }

    pthread_t threads[NR_CPU];
    args_t_js thread_args[NR_CPU];

    int per_core = combinations.size() / NR_CPU;
    int extra = combinations.size() % NR_CPU;


    for (int i = 0; i < NR_CPU; i++) {

        //create a sub vector containing the files to operate on for each worker thread.
        int init = i * per_core;
        int end = i != NR_CPU - 1 ? (i + 1) * per_core : ((i + 1) * per_core) + extra;
        std::vector<long> sub_vec(&combinations[init], &combinations[end]);

        thread_args[i].core = i;
        thread_args[i].sub_vec = sub_vec;

        pthread_create(&threads[i], NULL, thread_distances, &thread_args[i]);
    }

    for (int i = 0; i < NR_CPU; ++i)
        pthread_join(threads[i], NULL);

    return js_dist;
}




//// Utilities

/**
 * Right trim a string.
 * @param str
 */
void inline trim(std::string str) {
    str.erase(str.find_last_not_of("\n\r\t") + 1);
}


/**
 * Executes a command and retrieves the stdout.
 * @param cmd
 * @return
 */
std::string exec(const char *cmd) {
    std::array<char, 4096> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 4096, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}


/**
 * Splits string over a char delimiter.
 * @tparam Out
 * @param s
 * @param delim
 * @param result
 */
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


/**
 * Merge two integers in a long.
 * @param x
 * @param y
 * @return
 */
long inline merge_int(int x, int y) {
    long l = (((long) x) << 32) | (y & 0xffffffffL);
    return l;
}


/**
 * Splits a long into two integers.
 * @param l
 * @param x
 * @param y
 */
void inline split_long(long l, int *x, int *y) {
    *x = (int) (l >> 32);
    *y = (int) l;
}
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np


def laplace_smoothing(prob):
    smooth_prob = []
    n = len(prob)
    print('sum of probabilities', sum(prob))
    for p in prob:
        print(p)
        s_p = (p + 1) / (1 + n)
        smooth_prob.append(s_p)

    return np.asarray(smooth_prob)


def jensen_shannon_dist(vec1, vec2):

    prob1 = vec1 / norm(vec1, ord=1)
    prob2 = vec2 / norm(vec2, ord=1)

    s_prob1 = laplace_smoothing(prob1)
    s_prob2 = laplace_smoothing(prob2)
    print(s_prob1, s_prob2)

    m_vec = 0.5 * (s_prob1 + s_prob2)

    kl_1 = entropy(s_prob1, m_vec)
    kl_2 = entropy(s_prob2, m_vec)

    return 0.5 * (kl_1 + kl_2)


def compute_js_dist():
    vec1 = [0, 45678456, 1]
    vec2 = [0, 0, 0]

    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    print(jensen_shannon_dist(vec1, vec2))

if __name__ == '__main__':
    compute_js_dist()

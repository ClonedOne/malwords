from scipy.stats import entropy
import numpy as np
import math


def laplace_smoothing(vec):
    """
    Smoothing method also called Add-1 smoothing. 
    Transforms each word count into a non-zero probability. 
    
    :param vec: word count vector
    :return: non-zero probability vector
    """

    smooth_prob = []
    n = len(vec)
    s = sum(vec)
    for p in vec:
        s_p = (p + 1) / (s + n)
        smooth_prob.append(s_p)

    return np.asarray(smooth_prob)


def jensen_shannon_dist(prob1, prob2):
    """
    Computes the jensen-shannon distance as square root of the jensen-shannon divergence. 
    
    :param prob1: word probability vector
    :param prob2: word probability vector
    :return: jensen-shannon distance of the 2 word probability vectors
    """

    m_vec = (prob1 + prob2) * 0.5

    kl_1 = entropy(prob1, m_vec)
    kl_2 = entropy(prob2, m_vec)

    jd_divergence = 0.5 * (kl_1 + kl_2)

    return math.sqrt(jd_divergence)


def compute_js_dist(vec1, vec2):
    """
    Given two word count vectors computes the jensen-shannon distance.
    
    :param vec1: word count vector
    :param vec2: word count vector
    :return: jensen-shannon distance of the 2 word count vectors
    """

    vec1 = vec1.toarray().flatten()
    vec2 = vec2.toarray().flatten()

    vec1 = laplace_smoothing(vec1)
    vec2 = laplace_smoothing(vec2)

    dist = jensen_shannon_dist(vec1, vec2)

    print(dist)
    return dist

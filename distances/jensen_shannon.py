from scipy.stats import entropy
import numpy as np
import math


def jensen_shannon_dist(vec1, vec2):
    """
    Given two word count vectors computes the jensen-shannon distance as square root of the jensen-shannon divergence.
    The entropy function computes the Kullback–Leibler divergence with normalization if needed.
    The absolute continuity condition is granted by the fact that each vector element is positive.
    Therefore m_vec[i] = 0 iff vec1[i] = vec2[i] = 0.

    :param vec1: word count vector
    :param vec2: word count vector
    :return: jensen-shannon distance of the 2 word count vectors
    """

    vec1 = vec1.toarray().flatten()
    vec2 = vec2.toarray().flatten()

    m_vec = 0.5 * (vec1 + vec2)
    return math.sqrt(0.5 * (entropy(vec1, m_vec) + entropy(vec2, m_vec)))
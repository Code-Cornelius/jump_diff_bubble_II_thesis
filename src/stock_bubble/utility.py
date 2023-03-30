import numpy as np

logistic_function = lambda x: 1. / (1. + np.exp(-x))

# logistic_function_for_X = lambda vect_x, weights: logistic_function(sum(vect_x * weights)) # we weight each component
logistic_function_for_X = lambda vect_x, weights: sum(logistic_function(vect_x) * weights)  # we weight each component

p = 0.05
kernel_int_power_law = lambda j: np.maximum((1 - p) ** (j - 1) * p, 0)


def from_deltas_to_a_taus(vect_deltas):
    vect_deltas = np.array(vect_deltas)
    assert all(vect_deltas > 0), "Deltas need to be positive."
    # tau = 1/(1 - a)
    # a = 1 - 1/tau
    the_as_tau = 1. - 1. / vect_deltas
    return the_as_tau


def kernel_int_geom(arr, p_scale):
    # arr is a np vector.

    # approximation of an exponential distribution.
    # if p = 1/n and X is geometrically distributed with parameter p,
    # then the distribution of X/n approaches an exponential distribution with expected value 1
    return (1 - p_scale) ** (arr - 1) * p_scale

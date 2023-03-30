"""
Calculate the expected time of delay to reincorporate the delayed compensation as explained in the thesis.
The method used in this script is bisect.
"""
import math

import numpy as np
from scipy.optimize import bisect


def expect_lengths_seq_folded_normal(sigma_2, nb_rep_mc, length_cumsum):
    normal_folded = np.abs(np.random.normal(0., math.sqrt(sigma_2), (nb_rep_mc, length_cumsum)))
    normal_folded2 = np.cumsum(normal_folded, axis=1)
    lengths_mc = np.argmax(normal_folded2 >= 1, axis=1) + 1
    # be careful, from overflow:
    # Just a word of caution: if there's no True value in its input array,
    # np.argmax will happily return 0 (which is not what you want in this case)
    return np.mean(lengths_mc)


def expect_lengths_seq_folded_uniform(bounds, nb_rep_mc, length_cumsum):
    normal_folded = np.random.uniform(*bounds, (nb_rep_mc, length_cumsum))
    normal_folded2 = np.cumsum(normal_folded, axis=1)
    lengths_mc = np.argmax(normal_folded2 >= 1, axis=1) + 1
    # be careful, from overflow:
    # Just a word of caution: if there's no True value in its input array,
    # np.argmax will happily return 0 (which is not what you want in this case)
    return np.mean(lengths_mc)


nb_rep_mc = 10000
length_cumsum = 2000
targeted_time = 10
# func_to_root_find = lambda param: expect_lengths_seq_folded_normal(param, nb_rep_mc, length_cumsum) - targetted_time
func_to_root_find = lambda param: expect_lengths_seq_folded_uniform((0, param), nb_rep_mc,
                                                                    length_cumsum) - targeted_time

# sanity check
a = 0.002
b = 1.5
print(f"We search for the parameter on the interval a={a}, b={b}.")
# we need that the evaluation at a and b are of different signs.
print("f(a) = ", func_to_root_find(a))
print("f(b) = ", func_to_root_find(b))

# we replace the value of the number of points we want to use for computing the sequence
# depending on how many is required for the smallest value which will be used for evaluating the function (a).
# this is a good idea because the function is monotonically decreasing.
# we multiply the expected value by 5 so most of the cases (for Monte Carlo) are included.
length_cumsum = int(func_to_root_find(a) * 5)

print(f"The parameter in order to get E[T_d.c.] = {targeted_time} is equal to ",
      bisect(func_to_root_find, a, b))

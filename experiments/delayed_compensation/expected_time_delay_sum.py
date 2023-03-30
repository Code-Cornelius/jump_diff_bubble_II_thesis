"""
Calculate the expected time of delay to reincorporate the delayed compensation as explained in the thesis.
The method used in this script is Monte Carlo.
"""
import math

import numpy as np
import scipy
from corai_plot import APlot
from tqdm import tqdm


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


irwin_hall_cdf = lambda x, n: 1 / math.factorial(n) * \
                              np.sum(
                                  [pow(-1, k) * scipy.special.binom(n, k) * pow(x - k, n)
                                   for k in range(math.floor(x) + 1)
                                   ]
                              )
# will not work for x or n too big. Big means do not take x = 10, n = 150, otherwise pow(x,n) will give you 1E150.
irwin_hall_cdf = np.vectorize(irwin_hall_cdf)

xx = np.linspace(0, 10, 1000)
yy_1 = irwin_hall_cdf(xx, 1)
yy_2 = irwin_hall_cdf(xx, 2)
yy_4 = irwin_hall_cdf(xx, 4)
yy_8 = irwin_hall_cdf(xx, 8)

a = APlot()
a.uni_plot(0, xx, yy_1,
           dict_plot_param={"markersize": 0, "color": "b", "linewidth": 2, "label": "$n=1$"})
a.uni_plot(0, xx, yy_2,
           dict_plot_param={"markersize": 0, "color": "r", "linewidth": 2, "label": "$n=2$"})
a.uni_plot(0, xx, yy_4,
           dict_plot_param={"markersize": 0, "color": "orange", "linewidth": 2, "label": "$n=4$"})
a.uni_plot(0, xx, yy_8,
           dict_plot_param={"markersize": 0, "color": "green", "linewidth": 2, "label": "$n=8$"},
           dict_ax={"title": "Comparison Irwin-Hall distribution for different $n$.",
                    "xlabel": "x", "ylabel": "$P(\sum_{i=1}^n f_i < x)$"})
a.tight_layout()
a.show_legend()

scaling_factor_of_interval = 1.  # if = 2, it means we look at U_[0,2]
ET = np.sum([irwin_hall_cdf(1. / scaling_factor_of_interval, n) for n in range(400)])
print(f"E[T_n] for U_[0,{scaling_factor_of_interval}] is ", ET)

# section ######################################################################
#  #############################################################################
#  Plot the Expected time as a function of the interval

res = []
scales = np.linspace(0.05, 1., 100)
for scale in scales:
    probabilities_to_reach_1_for_n_samp = [irwin_hall_cdf(1. / scale, n) for n in range(400)]
    ans = np.sum(probabilities_to_reach_1_for_n_samp)
    res.append(ans)

b = APlot()
b.uni_plot(0, scales, res,
           dict_plot_param={"markersize": 4, "marker": "x",
                            "color": "b", "linewidth": 0},
           dict_ax={"title": "", "xlabel": "h where $f_i \sim U_{[0,h]}$, $X = \sum_{i=1}^n f_i$",
                    "ylabel": "$E[T_{X>1}]$"})
b.tight_layout()

# section ######################################################################
#  #############################################################################
#  NORMAL
sigma_2s = np.linspace(0.0001, 0.008, 70)
nb_rep_mc = 10000
length_cumsum = 300

expected_time_normal = [expect_lengths_seq_folded_normal(sigma_2, nb_rep_mc, length_cumsum) for sigma_2 in
                        tqdm(sigma_2s)]
c = APlot()
c.uni_plot(0, sigma_2s, expected_time_normal,
           dict_plot_param={"markersize": 4, "marker": "x",
                            "color": "b", "linewidth": 0},
           dict_ax={"title": "", "xlabel": "$\sigma^2$ where $f_i \sim | N(0,\sigma^2)| $, $X = \sum_{i=1}^n f_i$",
                    "ylabel": "$E[T_{X>1}]$ by Monte Carlo"})
c.tight_layout()

# section ######################################################################
#  #############################################################################
#  Uniform again

expected_time_uni = [expect_lengths_seq_folded_uniform((0, h), nb_rep_mc, length_cumsum) for h in tqdm(scales)]
d = APlot()
d.uni_plot(0, scales, expected_time_uni,
           dict_plot_param={"markersize": 4, "marker": "x",
                            "color": "b", "linewidth": 0},
           dict_ax={"title": "", "xlabel": "h where $f_i \sim U_{[0,h]}$, $X = \sum_{i=1}^n f_i$",
                    "ylabel": "$E[T_{X>1}]$ by Monte Carlo"})
d.tight_layout()

# a.save_plot("CDF_irwin")
# b.save_plot("expected_time_uniform_ana")
# c.save_plot("expected_time_normal_mc")
# d.save_plot("expected_time_uniform_mc")

APlot.show_plot()

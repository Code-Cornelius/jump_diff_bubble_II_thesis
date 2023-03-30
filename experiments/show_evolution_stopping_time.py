"""
Show the stopping times' evolution through time on a sampled Brownian path.
"""

import math

import numpy as np
from corai_plot import APlot

from src.stock_bubble.plot_bubbles import plot_add_phases
from src.stock_bubble.sample_self_exciting_bubble import sample_self_exciting_bubble
from src.stock_bubble.utility import kernel_int_geom

np.random.seed(142)

# section ######################################################################
#  #############################################################################
#  Parameters
T_years = 4
ndays_per_year = 250
T = T_years * ndays_per_year
delta_INVAR = 1  # in days
DAYS_TRADING = np.arange(0, T + 1, 1) / (ndays_per_year * delta_INVAR)

DELTAS_LOCAL_COMPUTING = [250]
WEIGHTS_DELTAS = [1. / len(DELTAS_LOCAL_COMPUTING)] * len(DELTAS_LOCAL_COMPUTING)
sigma_bar_ann = 0.04  # standard deviation p.a., corresponds to 0.0225 of volatility
# this is normal that we do not put higher values (like 0.5 giving 0.25 vol) bc we also have the jump part increasing the volatility.
sigma_bar = sigma_bar_ann / np.sqrt((ndays_per_year * delta_INVAR))  # std per day

# initial conditions
sigma0 = sigma_bar

kernel = lambda indices: kernel_int_geom(indices, 0.1)

# section ######################################################################
#  #############################################################################
#  Sampling
(S, S_bm, S_up_jumps, S_down_jumps,
 r, r_bm, r_up_jumps, r_down_jumps,
 sigma2, X, N, mispricing_intensity,
 lambda_t, list_regime_change,
 argmin_list, argmax_list,
 argmin_corresponding_times,
 argmax_corresponding_times) = \
    sample_self_exciting_bubble(T, (0, sigma0 * sigma0, 1., 0), np.array([0.0]), np.array([0.]), 0., 0., sigma_bar,
                                np.array([0.]), np.array([0.0, 0.0]), np.array([0.0, 0.0]),
                                np.array([[0.0, 0.0], [0.0, 0.]]), np.array([[0.0, 0.0], [0.0, 0.]]),
                                np.array([0.0, 0.0]), 0., 0., delta_INVAR, kernel, DELTAS_LOCAL_COMPUTING,
                                WEIGHTS_DELTAS, burnin=False, store_stop_time=True, use_delay_compensation=False,
                                remove_all_mispricing_mechanism=True)

# section ######################################################################
#  #############################################################################
#  Plotting

a = APlot(how=(1, 1), figsize=(7, 7))

a.uni_plot(0, xx=DAYS_TRADING * ndays_per_year * delta_INVAR, yy=np.log(S),
           dict_plot_param={"markersize": 0,
                            'label': 'Log Price Stock',
                            'color': 'blue'},
           dict_ax={'ylabel': 'Log price', 'xlabel': '', 'title': 'Sample Path with Phases '
                                                                  '(Green is Bubble, Red is Drawdown) \n'
                                                                  'with represented as crosses the local extremas '
                                                                  '(Purple is local max, Gold is local min).'})

print(list_regime_change)
# we need to multiply the values, since: look above, we also multiplied
list_regime_change[0, :] = list_regime_change[0, :] * ndays_per_year * delta_INVAR
print(list_regime_change)

plot_add_phases(a, list_regime_change, ndays_per_year, delta_INVAR, DAYS_TRADING * ndays_per_year * delta_INVAR,
                WEIGHTS_DELTAS)

# adding crosses and lines
for argimax, times_argimax in zip(argmax_list, argmax_corresponding_times):
    day = DAYS_TRADING[argimax] * ndays_per_year * delta_INVAR
    a.plot_point(x=day, y=math.log(S[argimax]),
                 nb_ax=0, dict_plot_param={"markersize": 4,
                                           "marker": "x",
                                           "label": "",
                                           'color': 'purple'})
    a.plot_line(a=0., b=math.log(S[argimax]), xx=times_argimax, nb_ax=0,
                dict_plot_param={"markersize": 0.2,
                                 "marker": "o",
                                 "linewidth": 1,
                                 "label": "",
                                 'color': 'purple'})
for argimin, times_argimin in zip(argmin_list, argmin_corresponding_times):
    day = DAYS_TRADING[argimin] * ndays_per_year * delta_INVAR
    a.plot_point(x=day, y=math.log(S[argimin]),
                 nb_ax=0, dict_plot_param={"markersize": 4,
                                           "marker": "x",
                                           "label": "",
                                           'color': 'gold'})
    a.plot_line(a=0., b=math.log(S[argimin]), xx=times_argimin, nb_ax=0,
                dict_plot_param={"markersize": 0,
                                 "linewidth": 1,
                                 "label": "",
                                 'color': 'gold'})

a.show_legend()
a.show_plot()

"""
We sample a bubble of type-II and observe the difference between the
 mispricing index's value when approximated by the difference equations, and when
 it is exactly computed.
"""
import numpy as np
from corai_plot import APlot

from src.stock_bubble.sample_self_exciting_bubble import sample_self_exciting_bubble
from src.stock_bubble.utility import from_deltas_to_a_taus, kernel_int_geom

np.random.seed(142)

# section ######################################################################
#  #############################################################################
#  Parameters

T_years = 5
ndays_per_year = 250
T = T_years * ndays_per_year
delta_INVAR = 1  # in days

DAYS_TRADING = np.arange(0, T + 1, 1) / (ndays_per_year * delta_INVAR)

DELTAS_LOCAL_COMPUTING = [250]

WEIGHTS_DELTAS = [1. / len(DELTAS_LOCAL_COMPUTING)] * len(DELTAS_LOCAL_COMPUTING)

# mispricing index
X_bar = -4
S_LOG_FCT = np.array([0.0003])  # used to be 0.00033; big s means harder to get 1 for L(X_t)

THE_AS_TAU = from_deltas_to_a_taus([100])
WEIGHTS_AS = np.array([1. / len(THE_AS_TAU)] * len(THE_AS_TAU))

# fixed parameter values
sigma_bar_ann = 0.15
# std per day
sigma_bar = sigma_bar_ann / np.sqrt((ndays_per_year * delta_INVAR))

# return
r_bar_ann = 0.00  # p.a.
R_BAR = (1 + r_bar_ann) ** (1 / (ndays_per_year * delta_INVAR)) - 1

# initial conditions
S0 = 1
sigma0 = sigma_bar
r0 = R_BAR
X0 = X_bar
initial_conditions = X0, sigma0 * sigma0, S0, R_BAR
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
 argmax_corresponding_times) = sample_self_exciting_bubble(T, initial_conditions, THE_AS_TAU, WEIGHTS_AS, X_bar, R_BAR,
                                                           sigma_bar, S_LOG_FCT, np.array([0.03, 0.1]),
                                                           np.array([0.1, 0.5]),
                                                           np.array([[0.7, 0.0], [0.0, 0.7]]),
                                                           np.array([[0.2, 0.0], [0.0, 0.1]]),
                                                           np.array([0.2, -0.8]) / 100., 0.0, 1.,
                                                           delta_INVAR, kernel, DELTAS_LOCAL_COMPUTING, WEIGHTS_DELTAS,
                                                           burnin=True, store_stop_time=False,
                                                           use_delay_compensation=False,
                                                           remove_all_mispricing_mechanism=False)

#### I compared the computation of X_t mispricing index directly by hand versus the different equations we had.
#### We find some difference, especially when mispricing increases a lot suddenly.
# compute X_t from the returns
tau = 100
xaaa = [1. / S_LOG_FCT[0] / tau * np.sum(r[i:i + tau + 1] - R_BAR) + X_bar for i in range(0, len(r) - tau)]
aplot = APlot(how=(1, 1))
aplot.uni_plot(0, xx=DAYS_TRADING[-len(xaaa):],
               yy=X.flatten()[-len(xaaa):],
               dict_plot_param={"markersize": 0,
                                'label': 'With Increments',
                                'color': 'blue'},
               dict_ax={'ylabel': '$X_t$', 'title': 'Comparison approx. and real $X_t$', 'xlabel': ''})
aplot.uni_plot(0, xx=DAYS_TRADING[-len(xaaa):], yy=xaaa,
               dict_plot_param={"markersize": 0,
                                'label': 'Real average',
                                'color': 'red'})
aplot.show_legend(0)
APlot.show_plot()

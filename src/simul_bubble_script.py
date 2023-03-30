import numpy as np
from corai_plot import APlot
from corai_util.tools.src.function_writer import factory_fct_linked_path

from root_dir import ROOT_DIR
from src.stock_bubble.plot_bubbles import plot_add_phases, \
    plot_four_timeseries_decomp, plot_bubble_2d
from src.stock_bubble.sample_self_exciting_bubble import sample_self_exciting_bubble
from src.stock_bubble.utility import from_deltas_to_a_taus, kernel_int_geom

np.random.seed(142)

# section ######################################################################
#  #############################################################################
#  Parameters

name_for_saving = ""

T_years = 10
ndays_per_year = 250
T = T_years * ndays_per_year
delta_invar = 1  # in days
DAYS_TRADING = np.arange(0, T + 1, 1) / (ndays_per_year * delta_invar)

# DELTAS_LOCAL_COMPUTING = [250, 750, 1500]
DELTAS_LOCAL_COMPUTING = [
    250]  # 15 is an influence of 3 weeks, 90 is an influence of 4.5 months, 250 is an influence of 1 y,

WEIGHTS_DELTAS = [1. / len(DELTAS_LOCAL_COMPUTING)] * len(DELTAS_LOCAL_COMPUTING)
# WEIGHTS_DELTAS = [0.25, 0.25, 1.] # over compensation

# mispricing index
X_bar = -4
S_LOG_FCT = np.array([0.0003, 0.0002, 0.0001])  # used to be 0.00033; big s means harder to get 1 for L(X_t)
CHARAC_TIME_A_TAUS = from_deltas_to_a_taus([100, 250, 750])
WEIGHTS_CHARAC_TIME_A_TAUS = np.array([1. / len(CHARAC_TIME_A_TAUS)] * len(CHARAC_TIME_A_TAUS))

print("The As Tau: ", CHARAC_TIME_A_TAUS)
A_TAU = 0.998
N_S = np.round((1 - A_TAU) / S_LOG_FCT, 3)
TAU = int(np.round(1 / (1 - A_TAU), 0))
print("Tau ", TAU)
print("n_s", N_S)

# fixed parameter values

# GARCH(1,1)
ALPHA_GARCH = 0.01
BETA_GARCH = 0.95

# standard deviation p.a., corresponds to 0.0225 of annual volatility,
# lower than observed but remember we add jumps on top, increasing the volatility.
# this is normal that we do not put higher values (like 0.5 giving 0.25 vol)
# bc we also have the jump part increasing the volatility.
SIGMA_BAR_ANN = 0.15
# SIGMA_BAR_ANN = 0.0

sigma_bar = SIGMA_BAR_ANN / np.sqrt((ndays_per_year * delta_invar))

# return
R_BAR_ANN = 0.00  # p.a.
R_BAR = (1 + R_BAR_ANN) ** (1 / (ndays_per_year * delta_invar)) - 1

# Hawkes part
nu_matrix = np.array([0.01,
                      0.05])
nu2_matrix = np.array([0.001,
                       0.2])

eta_matrix = np.array([[0.7, 0.0],
                       [0.0, 0.6]])
eta2_matrix = np.array([[0.1, 0.0],
                        [0.0, 0.35]])

# nu_matrix = np.array([0.0,
#                       0.0])  # nu+, nu-
# nu2_matrix = np.array([0.0,
#                        0.0])  # nu+, nu-
# eta_matrix = np.array([[0.0, 0.0],
#                        [0.0, 0.0]])
# eta2_matrix = np.array([[0.0, 0.0],
#                         [0.0, 0.0]])

kappa_matrix = np.array([0.5,
                         -2.]) / 100.  # values in percent

# Sanity checks
assert ((nu_matrix >= 0.) & (nu_matrix <= 1.)).all(), \
    "Nu needs to be smaller than 1 because it is combined with the logistic function of the mispricing index."
assert kappa_matrix[0] >= 0 and kappa_matrix[1] <= 0, \
    "You need that the first entry of kappa is positive, and second negative"
assert len(DELTAS_LOCAL_COMPUTING) == len(WEIGHTS_DELTAS), \
    "The lengths of DELTAS_LOCAL_COMPUTING and WEIGHTS_DELTAS need to be the same, " \
    "but got " + str(len(DELTAS_LOCAL_COMPUTING)) + " and " + str(len(WEIGHTS_DELTAS)) + "."
assert len(S_LOG_FCT) == len(CHARAC_TIME_A_TAUS) == len(WEIGHTS_CHARAC_TIME_A_TAUS), \
    "The lengths of S_LOG_FCT, CHARAC_TIME_A_TAUS and WEIGHTS_CHARAC_TIME_A_TAUS need to be the same, " \
    "but got " + str(len(S_LOG_FCT)) + " and " + str(len(CHARAC_TIME_A_TAUS)) + " and " + str(
        len(WEIGHTS_CHARAC_TIME_A_TAUS)) + "."

p_kernel_self_exc = 0.1

# initial conditions
S0 = 1
sigma0 = sigma_bar
r0 = R_BAR
X0 = X_bar
initial_conditions = X0, sigma0 * sigma0, S0, R_BAR
kernel = lambda indices: kernel_int_geom(indices, p_kernel_self_exc)

# section ######################################################################
#  #############################################################################
#  Sampling

(S, S_bm, S_up_jumps, S_down_jumps,
 r, r_bm, r_up_jumps, r_down_jumps,
 sigma2, X, N, mispricing_intensity,
 lambda_t, list_regime_change,
 argmin_list, argmax_list,
 argmin_corresponding_times, argmax_corresponding_times) \
    = sample_self_exciting_bubble(T, initial_conditions, CHARAC_TIME_A_TAUS, WEIGHTS_CHARAC_TIME_A_TAUS, X_bar, R_BAR,
                                  sigma_bar, S_LOG_FCT, nu_matrix, nu2_matrix, eta_matrix, eta2_matrix, kappa_matrix,
                                  ALPHA_GARCH, BETA_GARCH, delta_invar, kernel, DELTAS_LOCAL_COMPUTING, WEIGHTS_DELTAS,
                                  burnin=True, store_stop_time=False, use_delay_compensation=False,
                                  remove_all_mispricing_mechanism=False)

# section ######################################################################
#  #############################################################################
#  Plotting

# we average the value, so the decimal values are not too long on the plot.
parameters_to_put_on_plot = (
    np.round(X0, 4), np.round(sigma0 * sigma0, 4), np.round(S0, 4), np.round(r0, 6), CHARAC_TIME_A_TAUS,
    np.round(X_bar, 4), S_LOG_FCT, np.round(R_BAR_ANN, 6), np.round(SIGMA_BAR_ANN, 4),
    np.round(nu_matrix, 4), np.round(eta_matrix, 4), np.round(nu2_matrix, 4), np.round(eta2_matrix, 4),
    np.round(kappa_matrix, 4), ALPHA_GARCH, BETA_GARCH,
    delta_invar, kernel, DELTAS_LOCAL_COMPUTING)
mask = [int(0 * ndays_per_year * delta_invar),
        int(T_years * ndays_per_year * delta_invar + 1)]

plot1, plot2 = plot_bubble_2d(DAYS_TRADING, S, r, sigma2, X, N, mispricing_intensity, lambda_t, R_BAR_ANN,
                              parameters_to_put_on_plot, mask=mask, plot_only_price=False)

plot3, plot4 = plot_four_timeseries_decomp(DAYS_TRADING,
                                           S, S_bm, S_up_jumps, S_down_jumps,
                                           r, r_bm, r_up_jumps, r_down_jumps,
                                           mask=mask)
plot_add_phases(plot1, list_regime_change, ndays_per_year, delta_invar, DAYS_TRADING, WEIGHTS_DELTAS, mask=mask)
plot_add_phases(plot3, list_regime_change, ndays_per_year, delta_invar, DAYS_TRADING, WEIGHTS_DELTAS, mask=mask)

# section ######################################################################
#  #############################################################################
#  saving plots
linker = factory_fct_linked_path(ROOT_DIR, 'img')

#### CHANGE THIS NAME TO SAVE UNDER DIFFERENT NAMING####
# name = "_no_fct_eta"

plot1.save_plot(linker(["price_return_vol" + name_for_saving]))
plot2.save_plot(linker(["intensity_misprice" + name_for_saving]))
plot3.save_plot(linker(["decomp_4_ts" + name_for_saving]))
plot4.save_plot(linker(["returns_per_ts" + name_for_saving]))

PLOT_ACF_AND_SIGNAL_PROCESS = False
if PLOT_ACF_AND_SIGNAL_PROCESS:
    ############# AUTOCOVARIANCE
    import numpy as np
    import scipy.signal as sig
    from corai_plot import APlot

    signal = r

    sampled_frequencies_period, power_sde_period = sig.periodogram(signal, 1, scaling="density",
                                                                   return_onesided=False, window='hamming')

    freqs = np.fft.fftfreq(DAYS_TRADING.size, 1)
    idx = np.argsort(freqs)
    ps = np.abs(np.fft.fft(signal))  # rescale to get right norm
    ps *= ps

    sampled_frequencies_fft = freqs[idx]
    power_sde_fft = ps[idx]

    variance_period = np.sum(power_sde_period)
    print("Total Variance, for periodogram: ", variance_period)
    variance_fft = np.sum(ps)
    print("Total Variance, for fft :        ", variance_fft)

    # section ######################################################################
    #  #############################################################################
    #  autocorrelation sequence

    ######## compute the periodogram and the auto-covariance sequence.
    slicer = slice(len(signal) // 2 - 10000, len(signal) // 2 + 10000)
    auto_correl = np.correlate(signal[slicer], signal[slicer], mode='full')
    print('Auto correlation sequence and length: ', auto_correl, len(auto_correl))

    # section ######################################################################
    #  #############################################################################
    #  plot
    a = APlot(how=(3, 1))
    a.uni_plot(nb_ax=0, xx=DAYS_TRADING, yy=signal,
               dict_plot_param={'markersize': 0,
                                'color': 'r'},
               dict_ax={'ylabel': 'Intens. of Signal',
                        'title': '',
                        'xlabel': 'Time [seconds]'})
    a.uni_plot(nb_ax=1, xx=sampled_frequencies_fft, yy=power_sde_fft,
               dict_plot_param={'markersize': 0, 'linewidth': 1,
                                'color': 'g',
                                'label': 'fft'},
               dict_ax={'yscale': 'log',
                        'ylabel': 'power spectral density [V**2/Hz]',
                        'title': '',
                        'xlabel': 'frequency [Hz]',
                        'ylim': [1e-10, 1e4]})
    a._axs[-1].acorr(auto_correl, maxlags=len(auto_correl) // 2)
    a.set_dict_ax(nb_ax=2, dict_ax={'ylabel': 'Auto-correlation Sequence', 'title': '', 'xlabel': 'Lag $n$'})
    a.show_legend(1)
    a.tight_layout()

APlot.show_plot()

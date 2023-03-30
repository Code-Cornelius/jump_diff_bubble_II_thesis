import numpy as np
from corai_plot import APlot, AColorsetContinuous

from src.stock_bubble.general_routines import compute_weight_Wt


def plot_bubble_2d(days_trading, S, r, sigma2, X, N, mispricing_intensity,
                   lambda_t, r_bar_per_day, parameters_to_put_on_plot,
                   mask=None, plot_only_price=False):
    # mask is a tuple, with first and last elements you want to see
    if mask is not None:
        days_trading = days_trading[mask[0]: mask[1]]

        S = S[mask[0]: mask[1]]
        r = r[mask[0]: mask[1]]
        sigma2 = sigma2[mask[0]: mask[1]]
        X = X[:, mask[0]: mask[1]]
        N = N[:, mask[0]: mask[1]]
        mispricing_intensity = mispricing_intensity[mask[0]: mask[1]]
        lambda_t = lambda_t[:, mask[0]: mask[1]]

    # parameters to write on the plot
    (X_0, sigma2_0, S_0, r_0,
     the_as_tau, X_bar, s, r_bar_ann, sigma_bar_ann,
     mu_matrix, eta_matrix, mu2_matrix, eta2_matrix,
     kappa_val_jump_matrix,
     alpha_garch, beta_garch,
     delta_INVAR,
     kernel, delta_local_computing) = parameters_to_put_on_plot
    parameters = [X_0, sigma2_0,
                  np.round(the_as_tau, 4), X_bar, s, r_bar_ann, sigma_bar_ann,
                  # we round the_as_tau here so there is no numerical error lower, when we compute the taus
                  mu_matrix, mu2_matrix,
                  eta_matrix[0, 0], eta2_matrix[0, 0],
                  eta_matrix[1, 1], eta2_matrix[1, 1],
                  kappa_val_jump_matrix,
                  alpha_garch, beta_garch,
                  delta_INVAR, "Geom. kern.",
                  delta_local_computing]
    name_parameters = ["$X_0$", "$\\sigma^2_0$",
                       "$a_{\\tau}$", "$\\bar{X}$", "$s$", "$\\bar{r}_{ann}$", "$\\bar{\\sigma}_{ann}$",
                       "$d_{+}^1, d_{-}^1$", "$d_{+}^2, d_{-}^2$",
                       "$d_{+,+}^1$", "$d_{+,+}^2$",
                       "$d_{-,-}^1$", "$d_{-,-}^2$",
                       "$\\kappa$",
                       "$\\alpha_{GARCH}$", "$\\beta_{GARCH}$",
                       "$\Delta_{INVAR}$", "kernel",
                       "${}^{t}\Delta$"]

    if not plot_only_price:  # we create a APlot of different shape depending on condition and last step is to plot the price.
        REMOVE_VOL_PLOT = False
        a = APlot(how=(2 if REMOVE_VOL_PLOT else 3, 1), figsize=(7, 7), sharex=True)
        b = APlot(how=(2, 2), figsize=(7, 7), sharex=True)
        a.uni_plot(nb_ax=1, xx=days_trading, yy=r,
                   dict_plot_param={"markersize": 0, 'label': 'returns stock', 'color': 'blue'},
                   dict_ax={'ylabel': 'returns $r$', 'xlabel': 'Time (years)', 'title': ''})
        a.plot_line(nb_ax=1, a=0., b=-0.1, xx=days_trading,
                    dict_plot_param={"markersize": 0, "linestyle": "--", 'label': '', 'color': 'black'})
        a.plot_line(nb_ax=1, a=0., b=0.1, xx=days_trading,
                    dict_plot_param={"markersize": 0, "linestyle": "--", 'label': 'Returns = $ \pm 0.1$',
                                     'color': 'black'})

        a.plot_line(nb_ax=1, a=0., b=-0.05, xx=days_trading,
                    dict_plot_param={"markersize": 0, "linestyle": "-.", 'label': '', 'color': 'black'})
        a.plot_line(nb_ax=1, a=0., b=0.05, xx=days_trading,
                    dict_plot_param={"markersize": 0, "linestyle": "-.", 'label': 'Returns = $ \pm 0.05$',
                                     'color': 'black'})
        if not REMOVE_VOL_PLOT:
            a.uni_plot(nb_ax=2, xx=days_trading, yy=sigma2,
                       dict_plot_param={"markersize": 0, 'label': 'Volatility', 'color': 'blue'},
                       dict_ax={'ylabel': '$\sigma^2$', 'xlabel': '', 'title': ''})
    else:
        a = APlot(how=(1, 1))
        b = APlot(how=(2, 2), figsize=(7, 7), sharex=True)

    a.uni_plot(0, xx=days_trading, yy=np.log(S),
               dict_plot_param={"markersize": 0,
                                'label': 'Log Price Stock',
                                'color': 'blue'},
               dict_ax={'ylabel': 'Log price', 'xlabel': '', 'title': '',
                        'yint': True})
    a.uni_plot(0, xx=days_trading, yy=days_trading * r_bar_per_day + np.log(S[0]),
               dict_plot_param={"markersize": 0,
                                'label': 'Base return $\mu$',
                                'color': 'r',
                                'linestyle': '--'})

    ####################################################################################
    # we plot first lambda_t^- as it is bigger than the ^+
    b.uni_plot(0, xx=days_trading, yy=lambda_t[1],
               dict_plot_param={"markersize": 0,
                                'label': '$\lambda_t^-$',
                                'color': 'orange'},
               dict_ax={'ylabel': "$\lambda_t$", 'xlabel': '', 'title': ''
                        # , 'parameters': parameters, 'name_parameters': name_parameters
                        # no need for repetition if below activated
                        })
    b.uni_plot(0, xx=days_trading, yy=lambda_t[0],
               dict_plot_param={"markersize": 0,
                                'label': '$\lambda_t^+$',
                                'color': 'blue'})
    b.uni_plot(1, xx=days_trading, yy=N[1],
               dict_plot_param={"markersize": 0,
                                'label': "$N_t^-$",
                                'color': 'orange'})

    b.uni_plot(1, xx=days_trading, yy=N[0],
               dict_plot_param={"markersize": 0,
                                'label': "$N_t^+$",
                                'color': 'blue'},
               dict_ax={'ylabel': "$N_t$", 'xlabel': '', 'title': ''})

    b.uni_plot(2, xx=days_trading, yy=mispricing_intensity,
               dict_plot_param={"markersize": 0,
                                'label': '$L(X_t)$',
                                'color': 'blue'},
               dict_ax={'ylabel': '$L(X_t)$', 'title': '', 'xlabel': '',
                        'parameters': parameters, 'name_parameters': name_parameters})

    labels = ['$\\tau = $ ' + str(int(
        np.round(1. / (1. - a), 0)))
              for a in the_as_tau]  # create a label per time scale for mispricing
    if len(labels) == 1:
        labels = labels[0]  # when there is a single label in labels, it creates bugs.
    b.uni_plot(3, xx=days_trading, yy=X.transpose(),
               dict_plot_param={"markersize": 0,
                                'label': labels,
                                'color': None},
               # none color bc there are multiple lines, one per delta, so we want the color to be adaptive
               dict_ax={'ylabel': '$X_t$', 'title': '', 'xlabel': 'Time (years)'})

    a.show_legend()
    # a.tight_layout()
    b.show_legend()
    # b.tight_layout()

    ### price trajectory not log price.
    # price = APlot()
    # price.uni_plot(0, xx=days_trading, yy=S,
    #                dict_plot_param={"markersize": 0, 'label': 'Price Stock', 'color': 'blue'},
    #                dict_ax={'ylabel': 'Price', 'xlabel': '', 'title': ''})
    return a, b


def plot_add_phases(plot1, list_regime_change, ndays_per_year, delta_between_cnsec_times, days_trading, weights_deltas,
                    mask=None):
    # if you put a mask too big, the coloring will continue further than needed.

    # with mask, there is no double-checking that the colors agree with bubble and drawdown phases.
    if mask is not None:
        cdt = (list_regime_change[0, :] >= mask[0]) & (list_regime_change[0, :] <= mask[1])
        list_regime_change = list_regime_change[:, cdt].copy()
        # we add a last value, such that the colors stop before the end
        days_trading = days_trading.copy()
        days_trading[-1] = (mask[1] - 1) / (ndays_per_year * delta_between_cnsec_times)

    # defining the colors, this is a gradient from red to green.
    NB_COLORS = int(
        np.round(sum(weights_deltas), 5) * 100) + 1  # we take the number of colors by going from 0 to the total sum.
    # we round because sometimes there are numerical approximation making it truncate the number and it misses a color.
    # we go from 0. to 1. by 0.01 (included so 1 value more)
    COLORS = AColorsetContinuous('brg', NB_COLORS, (0.5, 0.9))[::-1]  # reverse ordering

    color_boxes = [0] * NB_COLORS  # 0 means the color has already been drawn
    # and we need to put the label. If there is a 1, no need for the label.
    # the label is related to the color.

    # iterating on the intervals
    ALPHA = 0.6
    dict_kwargs_axvspan = {'linestyle': '--', 'alpha': ALPHA, 'linewidth': 0.4}

    for i in range(len(list_regime_change[0, :])):
        beg_period = list_regime_change[0, i] / (ndays_per_year * delta_between_cnsec_times)
        if i == len(list_regime_change[0, :]) - 1:  # last iteration
            end_period = days_trading[-1]
        else:
            end_period = list_regime_change[0, i + 1] / (ndays_per_year * delta_between_cnsec_times)

        color, color_nb, weight_Wt = color_creat(COLORS, list_regime_change, i, weights_deltas)

        if not color_boxes[color_nb]:  # if we have not yet put the color on the plot, we add it.
            color_boxes[color_nb] = 1

            plot1._axs[0].axvspan(beg_period, end_period, facecolor=color,
                                  label="$\Omega_t =$ " + str(np.round(weight_Wt, 2)),
                                  **dict_kwargs_axvspan)
        else:
            plot1._axs[0].axvspan(beg_period, end_period, facecolor=color, **dict_kwargs_axvspan)

    # plot1.show_legend()  # we show legend so the legend of omega appears
    plot1.show_legend(loc='lower left')  # lower right bc price goes up with returns
    plot1.tight_layout()
    return


def color_creat(COLORS, list_regime_change, i, weights_deltas):
    list_flagDD_current = list_regime_change[1:, i]  # the flags
    weight_Wt = compute_weight_Wt(list_flagDD_current, weights_deltas)
    color_nb = int(
        np.round(weight_Wt, 2) * 100)  # we round the value of the weight, so it matches the 100 color grid.
    color = COLORS[color_nb]
    return color, color_nb, weight_Wt


def plot_four_timeseries_decomp(days_trading,
                                S, S_bm, S_up_jumps, S_down_jumps,
                                r, r_bm, r_up_jumps, r_down_jumps,
                                mask=None):
    prices = [S_up_jumps, S_down_jumps, S_bm]
    returns = [r_up_jumps, r_down_jumps, r_bm]
    colors = ['black', 'red', 'green', 'blue']
    labels = ['Full Stock (Log) Price', 'Positive Jumps Comp.', 'Negative Jumps Comp.', 'Brownian Motion Comp.']

    # mask is a tuple, with first and last elements you want to see
    if mask is not None:
        days_trading = days_trading[mask[0]: mask[1]]

        S = S[mask[0]: mask[1]]
        r = r[mask[0]: mask[1]]

    a = APlot(how=(1, 1), figsize=(7, 7))

    SHOW_BM_COMP = True

    b = APlot(how=(4 if SHOW_BM_COMP else 3, 1), figsize=(7, 7), sharex=True)

    for index_sample_path in range(3 if SHOW_BM_COMP else 2):
        if index_sample_path == 2:
            xlabel = 'Time (years)'
        else:
            xlabel = ''
        a.uni_plot(0, xx=days_trading, yy=np.log(prices[index_sample_path]),
                   dict_plot_param={"markersize": 0,
                                    'label': labels[index_sample_path + 1],
                                    'color': colors[index_sample_path + 1]},
                   dict_ax={'ylabel': 'Log price', 'xlabel': xlabel, 'title': ''})
        b.uni_plot(index_sample_path + 1, xx=days_trading, yy=returns[index_sample_path],
                   dict_plot_param={"markersize": 0, 'label': labels[index_sample_path + 1],
                                    'color': colors[index_sample_path + 1]},
                   dict_ax={'ylabel': 'returns', 'xlabel': xlabel, 'title': ''})

    a.uni_plot(0, xx=days_trading, yy=np.log(S),
               dict_plot_param={"markersize": 0, 'label': labels[0], 'color': colors[0]},
               dict_ax={'ylabel': 'Log price', 'xlabel': 'Time (years)', 'title': ''})

    b.uni_plot(0, xx=days_trading, yy=r,
               dict_plot_param={"markersize": 0, 'label': labels[0], 'color': colors[0]},
               dict_ax={'ylabel': 'returns', 'xlabel': '', 'title': ''})

    a.show_legend()
    b.show_legend()
    return a, b

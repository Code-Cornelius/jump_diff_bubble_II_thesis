"""
Visualization that both parametrisations are equivalent.
"""
import numpy as np
from corai_plot import *

a = APlot()

logistic_function = lambda x: 1. / (1. + np.exp(-x))

rho = 10.
eta = 0.7
epsilon = 0.3

xx = np.linspace(-20, 20, 1000)
yy = eta * (logistic_function(xx) + rho) / (1. + rho)
a.uni_plot(0, xx=xx, yy=yy,
           dict_plot_param={'marker': '', 'linewidth': 2, 'color': 'blue',
                            'label': "$\gamma  (L(X_t) + \\rho) / (1. + \\rho)$"})

yy = eta + epsilon * (1. - eta) * logistic_function(xx)
a.uni_plot(0, xx=xx, yy=yy,
           dict_plot_param={'marker': '', 'linewidth': 2, 'color': 'red',
                            'label': '$\\alpha + \\beta L(X_t)$'})

# section ######################################################################
#  #############################################################################
#  Lines
a.plot_line(nb_ax=0, xx=xx, a=0., b=eta,
            dict_plot_param={'marker': '', 'linewidth': 2, 'linestyle': ':', 'color': 'black',
                             'label': 'Bound $\\alpha'})

a.plot_line(nb_ax=0, xx=xx, a=0., b=eta * rho / (1 + rho),
            dict_plot_param={'marker': '', 'linewidth': 2, 'linestyle': '-.', 'color': 'black',
                             'label': 'Bound $\gamma \\rho / (1. + \\rho)$'})

a.plot_line(nb_ax=0, xx=xx, a=0., b=eta + epsilon * (1. - eta),
            dict_plot_param={'marker': '', 'linewidth': 2, 'linestyle': '--', 'color': 'black',
                             'label': 'Bound $\\alpha + \\beta$'})

a.set_dict_ax(0, {'title': '',
                  'xlabel': '',
                  'ylabel': ''})
a.show_legend()

# section ######################################################################
#  #############################################################################
#  Searching for same characterisation
a = APlot()

rho1 = 10.
eta1 = 0.7

xx = np.linspace(-20, 20, 1000)
yy = eta1 * (logistic_function(xx) + rho1) / (1. + rho1)
a.uni_plot(0, xx=xx, yy=yy,
           dict_plot_param={'marker': '', 'linewidth': 2, 'color': 'blue',
                            'label': f"$\\gamma  (L(X_t) + \\rho) / (1. + \\rho), "
                                     f"\\gamma = {eta1}, \\rho = {rho}$."})

eta2 = eta1 * rho1 / (1. + rho1)
epsilon2 = 0.175 * (1. - eta2)

yy = eta2 + epsilon2 * logistic_function(xx)
a.uni_plot(0, xx=xx, yy=yy,
           dict_plot_param={'marker': '', 'linewidth': 2,
                            'linestyle': '--',
                            'color': 'red', 'label':
                                f"$\\alpha + \\beta L(X_t), "
                                f"\\alpha = {np.round(eta2, 2)}, \\beta = {np.round(epsilon2, 3)}$"})

a.set_dict_ax(0, {'title': '',
                  'xlabel': '',
                  'ylabel': ''})
a.show_legend()
a.show_plot()

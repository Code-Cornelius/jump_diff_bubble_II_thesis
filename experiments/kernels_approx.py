"""
Visual comparison of the different kernels we could be using, in particular the powerlaw kernel and geometric.
Presents the approximation errors of certain kernels choices.
"""
import numpy as np
import scipy
from corai_plot import APlot

# assuming we want to use a kernel K for HP simulations/estimations.
# now, we approximate HP with an INAR(p). How good is this approximation?

# the powerlaw kernel. Its integral for p > 0 is  a/p.
alpha = 0.8
beta = 0.7
p = 0.8

lanbda = 0.5
pwl_k = lambda x: alpha * beta / np.power(1 + beta * x, 1. + p) / (alpha / (p))  # normalising factor
geom_k = lambda x: np.power(1 - lanbda, np.floor(x)) * lanbda
exp_k = lambda x: lanbda * np.exp(- lanbda * x)

# proxy_to_pwl = lambda x: alpha * beta / np.power(1 + beta * np.floor(x+0.5), 1. + p) / (alpha / (p))  # normalising factor
proxy_to_pwl = lambda xx: np.array([scipy.integrate.quad(pwl_k, np.floor(x), np.floor(x + 1.))[0] for x in xx])

range_plot = np.linspace(0, 10, 100000)
plot = APlot(how=(2, 1))

# pdf
plot.uni_plot(0, xx=range_plot, yy=pwl_k(range_plot),
              dict_plot_param={'label': 'pwl', 'color': 'b', 'markersize': 0, 'linewidth': 2},
              dict_ax={'title': 'Comparison different PDF', 'xlabel': 'Time since Event', 'ylabel': 'Kernel Value'})
plot.uni_plot(0, xx=range_plot, yy=geom_k(range_plot),
              dict_plot_param={'label': 'geom_k', 'color': 'g', 'markersize': 0, 'linewidth': 2})
plot.uni_plot(0, xx=range_plot, yy=exp_k(range_plot),
              dict_plot_param={'label': 'exp_k', 'color': 'r', 'markersize': 0, 'linewidth': 2})

plot.uni_plot(0, xx=range_plot, yy=proxy_to_pwl(range_plot),
              dict_plot_param={'label': 'pwl approx', 'color': 'm', 'markersize': 0, 'linewidth': 2})

# cdf
integration = lambda vect: np.cumsum(vect) * (range_plot[-1] + range_plot[0]) / len(range_plot)
plot.uni_plot(1, xx=range_plot, yy=integration(pwl_k(range_plot)),
              dict_plot_param={'label': 'pwl', 'color': 'b', 'markersize': 0, 'linewidth': 2},
              dict_ax={'title': 'Comparison different CDF', 'xlabel': 'Time since Event', 'ylabel': 'Kernel Value'})
plot.uni_plot(1, xx=range_plot, yy=integration(geom_k(range_plot)),
              dict_plot_param={'label': 'geom_k', 'color': 'g', 'markersize': 0, 'linewidth': 2})
plot.uni_plot(1, xx=range_plot, yy=integration(exp_k(range_plot)),
              dict_plot_param={'label': 'exp_k', 'color': 'r', 'markersize': 0, 'linewidth': 2})

plot.uni_plot(1, xx=range_plot, yy=integration(proxy_to_pwl(range_plot)),
              dict_plot_param={'label': 'pwl approx', 'color': 'm', 'markersize': 0, 'linewidth': 2})
plot.show_legend()
plot.tight_layout()

pwl_k = lambda x: p * beta ** p / np.power((x > beta) * x, 1. + p)
exp_k = lambda x: lanbda * np.exp(- lanbda * x)

range_plot = np.linspace(0, 20, 100000)
plot = APlot(how=(1, 1))

# pdf
plot.uni_plot(0, xx=range_plot, yy=pwl_k(range_plot),
              dict_plot_param={'label': 'pwl', 'color': 'b', 'markersize': 0, 'linewidth': 2},
              dict_ax={'title': 'Comparison different PDF', 'xlabel': 'Time since Event', 'ylabel': 'Kernel Value'})
plot.uni_plot(0, xx=range_plot, yy=exp_k(range_plot),
              dict_plot_param={'label': 'exp_k', 'color': 'r', 'markersize': 0, 'linewidth': 2})

plot.show_legend()
plot.tight_layout()

plot.show_plot()

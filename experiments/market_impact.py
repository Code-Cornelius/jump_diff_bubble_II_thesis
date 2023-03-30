"""
Market impact plots for the thesis.
Plot showing the shape of market impact as calculated by Rosenbaum et al. in Paul Jusselin
and Mathieu Rosenbaum. “No-arbitrage implies power-law market impact and rough volatility”.
"""
import numpy as np

import scipy.integrate
from corai_plot import APlot
from tqdm import tqdm

tt = np.linspace(0.0001, 4., 10001)
s = 1.


def market_impact(alpha, s, tt):
    res_func = []
    for t in tqdm(tt):
        cst = 1 - alpha
        function_to_int = lambda x: cst * 1. * (t - x <= s) * np.power(x, -alpha)
        int = scipy.integrate.quad(function_to_int, a=0, b=t)[0]  # takes back only real part

        # code by hand:
        # nodes_quad = np.arange(0.0001,t, tt[1] - tt[0] / 1000.) # we use arange so the nb of points used for each quadrature is
        # proportional to the length of the interval
        # using 0.001 in order to not have infinity.

        # f = 1. * (t - nodes_quad <= s) # equal to 1 when nodes_quad is smaller than s.
        # u_pow = np.power(nodes_quad, - alpha)
        # print("\ntime ", t)
        # print("x :", nodes_quad)
        # print("y :", f * u_pow)

        # int = scipy.integrate.simpson(f * u_pow, nodes_quad, even = 'first')

        res_func.append(int)
    return res_func


res_05 = market_impact(0.5, s, tt)
res_001 = market_impact(0.01, s, tt)
res_08 = market_impact(0.8, s, tt)

# print(res_func)
aplot = APlot()
aplot.uni_plot(0, tt, res_001, dict_plot_param={"color": "r", "marker": "", "linewidth": 3,
                                                "label": f"MI For $\\alpha =$ {0.01}."})
aplot.uni_plot(0, tt, res_05, dict_plot_param={"color": "b", "marker": "", "linewidth": 3,
                                               "label": f"MI For $\\alpha =$ {0.5}."})

aplot.uni_plot(0, tt, res_08, dict_plot_param={"color": "g", "marker": "", "linewidth": 3,
                                               "label": f"MI For $\\alpha =$ {0.8}."},
               dict_ax={"title": "", "xlabel": "", "ylabel": ""})

aplot.show_legend()

APlot.show_plot()

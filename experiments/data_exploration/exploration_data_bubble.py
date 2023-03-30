"""
Explore data about the stock market by plotting the returns and the prices of some indices.
Add insights on the particular HSI index.
"""
import math

import numpy as np
import pandas as pd
from corai_plot import APlot
from corai_util.tools.src.function_writer import factory_fct_linked_path

from root_dir import ROOT_DIR
from src.stock_bubble.stat_comput import print_ess_stat_returns

linker = factory_fct_linked_path(ROOT_DIR, 'data/daily_prices')
dataset_naming_list = ['dailyCAC40.csv',
                       'dailyDAX.csv',
                       'dailyDJIA.csv',
                       'dailyEuroStoxx600.csv',
                       'dailyHSI.csv',
                       'dailyNasdaq.csv',
                       'dailyNikkei.csv',
                       'dailySP500.csv']
dataset_addres_list = [linker([name]) for name in dataset_naming_list]

plot = APlot(how=(3, 3))
plot_returns = APlot(how=(3, 3))
for i in range(8):
    print("")
    print("            ###############   Dataset", dataset_naming_list[i], "   ###############")
    print("Analyse.")
    df = pd.read_csv(dataset_addres_list[i])
    times = np.array(df['t'].to_numpy())
    first_year = int(times[0][-4:])
    times = np.arange(0, len(times), 1) / 250 + int(times[0][-4:])  # convert it into numbers
    stock_val = df['S'].to_numpy()
    log_price = np.log(stock_val)

    returns = np.diff(log_price)
    print("Stats about returns:")
    print_ess_stat_returns(returns, times[-1], ndays_per_year=250)

    plot.uni_plot(nb_ax=i, xx=times, yy=stock_val,
                  dict_plot_param={'color': 'blue',
                                   'linestyle': 'solid',
                                   'linewidth': 0.5,
                                   'marker': "",
                                   'label': "Price"},
                  dict_ax={'title': '',
                           'xlabel': '',
                           'ylabel': ''})

    plot.uni_plot_ax_bis(nb_ax=i, xx=times, yy=log_price,
                         dict_plot_param={'color': 'red',
                                          'linestyle': 'solid',
                                          'linewidth': 0.5,
                                          'marker': "",
                                          'label': "Log Price"},
                         dict_ax={'ylabel': ''})
    plot._axs_bis[i].grid(True)
    log_prices_int_ticks = range(math.floor(min(log_price)) - 1,
                                 math.ceil(max(log_price)) + 1)
    plot._axs_bis[i].set_yticks(log_prices_int_ticks)  # forcing ticks to be integers

    plot_returns.uni_plot(nb_ax=i, xx=times[:-1], yy=returns,
                          dict_plot_param={'color': 'blue',
                                           'linestyle': 'solid',
                                           'linewidth': 0.5,
                                           'marker': "",
                                           'label': "Log Returns"},
                          dict_ax={'title': '',
                                   'xlabel': '',
                                   'ylabel': ''})
for nb_a in [6, 7]:
    for pl in [plot, plot_returns]:
        pl.set_dict_ax(nb_a, {'title': '',
                              'xlabel': 'Time in Years',
                              'ylabel': ''})
        pl.set_dict_ax(8, {'title': '',
                           'xlabel': '',
                           'ylabel': ''})
plot.show_legend()

# section ######################################################################
#  #############################################################################
#  Plot the Stock

plot = APlot(how=(1, 1))
df = pd.read_csv(dataset_addres_list[4])
times = np.array(df['t'].to_numpy())
times = np.arange(0, len(times), 1) / 250  # convert it into numbers
times += 1964.6
stock_val = df['S'].to_numpy()
log_price = np.log(stock_val)

plot.uni_plot_ax_bis(nb_ax=0, xx=times, yy=stock_val,
                     dict_plot_param={'color': 'blue',
                                      'linestyle': 'solid',
                                      'linewidth': 0.5,
                                      'marker': "",
                                      'label': "Price"},
                     dict_ax={'title': '',
                              'xlabel': 'Time in Years',
                              'ylabel': ''})

plot.uni_plot(nb_ax=0, xx=times, yy=log_price,
              dict_plot_param={'color': 'red',
                               'linestyle': 'solid',
                               'linewidth': 0.5,
                               'marker': "",
                               'label': "Log Price"},
              dict_ax={'ylabel': ''})
plot.set_dict_ax(0, {'title': '',
                     'xlabel': '',
                     'ylabel': ''})

plot.plot_line(a=0.15, b=-0.15 * 1930.5,
               xx=times,
               dict_plot_param={'color': 'black',
                                'linestyle': 'dotted',
                                'linewidth': 1.,
                                'marker': "",
                                'label': "First Trend"})
plot.plot_vertical_line(x=1998,
                        yy=np.array(plot.get_y_lim(nb_ax=0)),
                        nb_ax=0, dict_plot_param={'color': 'black',
                                                  'linestyle': '-',
                                                  'linewidth': 0.7,
                                                  'marker': "",
                                                  'label': "Change of Regime"})

plot.plot_line(a=0.04, b=-0.04 * 1760,
               xx=times,
               dict_plot_param={'color': 'black',
                                'linestyle': '-.',
                                'linewidth': 1.,
                                'marker': "",
                                'label': "Second Trend"})
plot.show_legend()

# section ######################################################################
#  #############################################################################
#  Plot the Bubbles

df = pd.read_csv(dataset_addres_list[4])
times = np.array(df['t'].to_numpy())
times = np.arange(0, len(times), 1) / 250  # convert it into numbers
times += 1964.
log_price = np.log(stock_val)

beg = 6000
end = 5000
times = times[beg:-end]
log_price = log_price[beg:-end]

plot = APlot(how=(3, 1))

interval_times = [320, 421, 999, 1136, 1186, 1288]
bs_values = [9.04, 9.4, 9.]
for i in range(3):
    plot.uni_plot(nb_ax=i, xx=times, yy=log_price,
                  dict_plot_param={'color': 'red',
                                   'linestyle': 'solid',
                                   'linewidth': 0.5,
                                   'marker': "",
                                   'label': "Log Price"},
                  dict_ax={'ylabel': ''})
    plot.set_dict_ax(i, {'title': 'daily HSI',
                         'xlabel': '',
                         'ylabel': ''})
    plot.plot_line(a=0., b=bs_values[i], xx=times[interval_times[2 * i]:interval_times[2 * i + 1]], nb_ax=i,
                   dict_plot_param={'color': 'black',
                                    'linestyle': '--',
                                    'linewidth': 1.,
                                    'marker': "",
                                    'label': "Characteristic Time Scale, length " +
                                             str(interval_times[2 * i + 1] - interval_times[2 * i])})
plot.show_legend()

# section ######################################################################
#  #############################################################################
#  second example

plot = APlot(how=(3, 1))

interval_times = [274, 321,
                  1051, 1099,
                  1135, 1186]
bs_values = [9.04, 9.57, 9.15]
for i in range(3):
    plot.uni_plot(nb_ax=i, xx=times, yy=log_price,
                  dict_plot_param={'color': 'red',
                                   'linestyle': 'solid',
                                   'linewidth': 0.5,
                                   'marker': "",
                                   'label': "Log Price"},
                  dict_ax={'ylabel': ''})
    plot.set_dict_ax(i, {'title': 'daily HSI',
                         'xlabel': '',
                         'ylabel': ''})
    plot.plot_line(a=0., b=bs_values[i], xx=times[interval_times[2 * i]:interval_times[2 * i + 1]], nb_ax=i,
                   dict_plot_param={'color': 'black',
                                    'linestyle': '--',
                                    'linewidth': 1.,
                                    'marker': "",
                                    'label': "Characteristic Time Scale, length " +
                                             str(interval_times[2 * i + 1] - interval_times[2 * i])})

plot.show_legend()
APlot.show_plot()

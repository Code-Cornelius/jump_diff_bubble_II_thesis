"""
Plot the bubble and crash of 1987 of the CAC40 index.
"""
import numpy as np
import pandas as pd
from corai_plot import APlot
from corai_util.tools.src.function_writer import factory_fct_linked_path

from root_dir import ROOT_DIR

linker = factory_fct_linked_path(ROOT_DIR, 'data/daily_prices')

df = pd.read_csv(linker(['dailyCAC40.csv']))
times = np.array(df['t'].to_numpy())
times = np.arange(0, len(times), 1) / 250. + 1969
stock_val = df['S'].to_numpy()
log_price = np.log(stock_val)

min_index = 4100
max_index = 4800
times = times[min_index:max_index]
stock_val = stock_val[min_index:max_index]
log_price = log_price[min_index:max_index]

plot = APlot(how=(1, 1))
plot.uni_plot(nb_ax=0, xx=times, yy=stock_val,
              dict_plot_param={'color': 'blue',
                               'linestyle': 'solid',
                               'linewidth': 2.,
                               'marker': "",
                               'label': "Price"},
              dict_ax={'title': "CAC40 Bubble and Crash of 1987",
                       'xlabel': 'Time in Years',
                       'ylabel': 'Price'})

# plotting the first line
plot.plot_line(a=235, b=-235 * 1980.5,
               xx=times,
               dict_plot_param={'color': 'black',
                                'linestyle': 'dotted',
                                'linewidth': 1.5,
                                'marker': "",
                                'label': "$\\tilde{S}$"})
plot.plot_vertical_line(x=1987.61,
                        yy=np.array(plot.get_y_lim(nb_ax=0)),
                        nb_ax=0, dict_plot_param={'color': 'red',
                                                  'linestyle': '-',
                                                  'linewidth': 1.5,
                                                  'marker': "",
                                                  'label': "$\\tau_{drawdown}$"})
plot.plot_line(a=105, b=-105 * 1978,
               xx=times,
               dict_plot_param={'color': 'black',
                                'linestyle': '-.',
                                'linewidth': 1.5,
                                'marker': "",
                                'label': "$D$"})
plot.show_legend()
APlot.show_plot()

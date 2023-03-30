"""
This script is used to identify the phases of the stock market. Here, we show
the phases for the CAC40.
"""
import math

import numpy as np
import pandas as pd
from corai_plot import APlot
from corai_util.tools.src.function_writer import factory_fct_linked_path
from tqdm import tqdm

from root_dir import ROOT_DIR
from src.stock_bubble.plot_bubbles import plot_add_phases
from src.stock_bubble.updater_bubble_flags import Updater_bubble_flags

linker = factory_fct_linked_path(ROOT_DIR, 'data/daily_prices')
dataset_naming_list = ['dailyCAC40.csv',
                       'dailyDAX.csv',
                       'dailyDJIA.csv',
                       'dailyEuroStoxx600.csv',
                       'dailyHSI.csv',
                       'dailyNasdaq.csv',
                       'dailyNikkei.csv',
                       'dailySP500.csv']
dataset_addres = linker([dataset_naming_list[0]])

plot = APlot(how=(1, 1))
df = pd.read_csv(dataset_addres)
times = np.array(df['t'].to_numpy())
first_year = int(times[0][-4:])
# we do not scale in the end so we can write the regimes
# times = np.arange(0, len(times), 1) / 250 + int(times[0][-4:])  # convert it into numbers
times = np.arange(0, len(times), 1) / 250  # convert it into numbers
stock_val = df['S'].to_numpy()
log_price = np.log(stock_val)

plot.uni_plot(nb_ax=0, xx=times, yy=log_price,
              dict_plot_param={'color': 'red',
                               'linestyle': 'solid',
                               'linewidth': 0.5,
                               'marker': "",
                               'label': "Log Price"},
              dict_ax={'title': '',
                       'xlabel': '',
                       'ylabel': ''})
plot._axs[0].grid(True)
log_prices_int_ticks = range(math.floor(min(log_price)) - 1,
                             math.ceil(max(log_price)) + 1)
plot._axs[0].set_yticks(log_prices_int_ticks)  # forcing ticks to be integers
plot.show_legend()

# section ######################################################################
#  #############################################################################
#  Analysis of regimes
DELTAS_LOCAL_COMPUTING = [50, 100, 250, 750]
WEIGHTS_DELTAS = [1. / len(DELTAS_LOCAL_COMPUTING)] * len(DELTAS_LOCAL_COMPUTING)

updater_bubble_flags = Updater_bubble_flags(DELTAS_LOCAL_COMPUTING)
list_flagDD = [False] * len(DELTAS_LOCAL_COMPUTING)
list_regime_change = [[0] + list_flagDD.copy()]  # at time zero, you start with regime 0

for t_index in tqdm(range(1, len(times) + 1)):
    (argmin_current, argmax_current, index_time_minus_delta) = \
        updater_bubble_flags.update_flagBubble(stock_val, list_flagDD, list_regime_change, t_index)

# converts the vector of flag whether is in bubble mode or not.
list_regime_change = np.array(list_regime_change,
                              dtype=int).transpose()  # line one gives time, line two gives regimes

plot_add_phases(plot, list_regime_change, 250, 1, times, WEIGHTS_DELTAS)

APlot.show_plot()

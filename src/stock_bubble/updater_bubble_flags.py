import math
import typing

import numpy as np


class Updater_bubble_flags(object):
    """
    This class is used to update the bubble flags, a describe in the algorithm of the paper.
    The class helps keeping track of the two different types of source of regime change: delta^s and delta^t.
    """

    # Minimal period for each flag before changing regime.
    DIVIDER_PERIOD = 2
    # the threshold of average realised standard deviation per day we require before changing the flag
    # A value of 0.01 corresponds to a threshold for the vol of 0.025 per year
    STD_EXCURSION = 0.01

    # WIP is it important that the ordering is kept?
    def __init__(self, deltas_local_computing):
        self.deltas_local_computing = deltas_local_computing
        self.times_per_ts_last_change = []
        self.minimal_periods = [delta // Updater_bubble_flags.DIVIDER_PERIOD
                                for delta in self.deltas_local_computing]

        # we create a list of the last time
        # a change to the flag happened. Hence, it is length list_flagDD and we put an outrageous low value
        # so it is always very low.
        self.times_per_ts_last_change = [-max(deltas_local_computing)] * len(deltas_local_computing)

    def update_flagBubble(self, S, list_flagDD: typing.Iterable, list_regime_change: typing.Iterable,
                          current_time: int):
        # flagBubble_vect, should not be only true and false, but should also be a number. The number of
        for i, delta_local_computing in enumerate(self.deltas_local_computing):
            index_time_minus_delta = max(current_time - delta_local_computing,
                                         0)  # max bc we do not want the index to be negative

            slice_price = S[index_time_minus_delta:current_time]
            argmin_price_over_delta_time = np.argmin(slice_price)
            argmax_price_over_delta_time = np.argmax(slice_price)

            # inside list_flagDD:
            # True means that for a timescale, we are in drawdown
            # False means that for a timescale, we are in bubble (not in drawdown)

            # #### Original Pattern, before D.S. suggestion
            # if argmax_price_over_delta_time < argmin_price_over_delta_time:
            #     list_flagDD[i] = True
            #
            # else: # elif argmin_price_over_delta_time < argmax_price_over_delta_time:
            #     list_flagDD[i] = False

            #### Another suggestion Excursion MECHANISM
            THRESHOLD = math.exp(math.sqrt(delta_local_computing) * Updater_bubble_flags.STD_EXCURSION)
            if argmax_price_over_delta_time < argmin_price_over_delta_time:  # potentially we are in a drawdown: prices go down
                # Now we are checking for huge variation in price
                if slice_price[-1] / \
                        slice_price[
                            argmin_price_over_delta_time] <= THRESHOLD:  # if price has increased less than a threshold:
                    list_flagDD[i] = True
                else:  # the variation has been too big and it means that switched to a bubble regime
                    list_flagDD[i] = False

            else:  # elif argmin_price_over_delta_time < argmax_price_over_delta_time: we are in a drawup
                # Now we are checking for huge variation in price
                if slice_price[argmax_price_over_delta_time] / \
                        slice_price[-1] >= THRESHOLD:  # if price has decreased more than a threshold:
                    list_flagDD[i] = True
                else:  # the variation has been too big and it means that we switched to a bubble regime
                    list_flagDD[i] = False

        ############# Minimum regime period
        # we are checking if there were changes since last regime change.
        # If yes, we check how long it has been, because we have enforced a minimal regime period.
        previous_list_flagDD = list_regime_change[-1][1:]
        # for i in range(len(self.deltas_local_computing)):
        #     statut_flag = list_flagDD[i]
        #     prev_stat_flag = previous_list_flagDD[i]  # we only check the last logged status,
        #     # if there is a change this round, it is a change compared to the last logged.
        #
        #     last_time_change = self.times_per_ts_last_change[i]
        #     minimal_period = self.minimal_periods[i]
        #     if statut_flag != prev_stat_flag:  # there has been a change:
        #         # check how long has it been since last change for this flag
        #         if current_time - last_time_change < minimal_period:  # if last change happened less than minimal period ago, we stick to it.
        #             list_flagDD[i] = prev_stat_flag  # we put the previous state back in the list of flags.
        #         else:
        #             self.times_per_ts_last_change[i] = current_time  # we change last time a change has been made

        # Storing
        if list_flagDD != previous_list_flagDD:  # if there is a regime change
            list_regime_change.append([current_time] + list_flagDD)

        return argmin_price_over_delta_time, argmax_price_over_delta_time, index_time_minus_delta

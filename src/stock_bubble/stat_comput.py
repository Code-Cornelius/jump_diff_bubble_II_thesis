import numpy as np


def separate_list_bubble_drawdown(list_elmnt, starts_with_bubble):
    if starts_with_bubble:
        list_elmnt_bub = list_elmnt[::2]
        list_elmnt_dra = list_elmnt[1::2]
    else:
        list_elmnt_bub = list_elmnt[1::2]
        list_elmnt_dra = list_elmnt[::2]
    return list_elmnt_bub, list_elmnt_dra


def print_avrg_std(list_elmnt, string):
    print(string, np.round(np.mean(list_elmnt), 2), " (+/-", np.round(np.std(list_elmnt), 2), ")", sep='')


def print_ess_stat_bub(flagBubble_vect, starts_with_bubble, R_BAR, DAYS_TRADING, ndays_per_year, delta_INVAR, S):
    length_time_regimes = np.diff(flagBubble_vect)

    # separating the two regimes lengths
    (length_time_regimes_bubble,
     length_time_regimes_drawdo) = separate_list_bubble_drawdown(length_time_regimes, starts_with_bubble)
    print_avrg_std(length_time_regimes_bubble, "Average duration of drawups   : ")
    print_avrg_std(length_time_regimes_drawdo, "Average duration of drawdowns : ")

    # we compute the % change between the beginning of a drawup and the end of a drawup (resp. drawdown).
    # also the % for the discounted price
    S_discounted = S * np.power((1 + R_BAR), - (
            DAYS_TRADING * ndays_per_year * delta_INVAR + 10000))  # Be careful, constant 10000 here corresponding to the burn in period from sampling.

    ratio_increase = []
    ratio_disc_increase = []
    for beg, end in zip(flagBubble_vect[:-1], flagBubble_vect[1:]):
        ratio_increase.append(S[end] / S[beg])
        ratio_disc_increase.append(S_discounted[end] / S_discounted[beg])

    # in ratio_increase, each pair of successive elements represents drawup/drawdown pair.
    ratio_increase_bubble, ratio_increase_drawdo = separate_list_bubble_drawdown(ratio_increase, starts_with_bubble)
    (ratio_disc_increase_bubble,
     ratio_disc_increase_drawdo) = separate_list_bubble_drawdown(ratio_disc_increase, starts_with_bubble)

    print_avrg_std(ratio_increase_bubble, "Average ratio increase during drawups              : ")
    print_avrg_std(ratio_increase_drawdo, "Average ratio increase during drawdowns            : ")
    print_avrg_std(ratio_disc_increase_bubble, "Average discounted ratio increase during drawups   : ")
    print_avrg_std(ratio_disc_increase_drawdo, "Average discounted ratio increase during drawdowns : ")


def print_ess_stat_returns(r, T_years, ndays_per_year):
    crash_values = [-0.01, -0.05, -0.1, -0.2]
    expected_rates = [T_years * ndays_per_year // 10.,
                      T_years,
                      int(T_years / 12.),
                      int(T_years / 35.)]
    # every two working weeks, every year, 12 years and 35 years.
    for crash_value, crash_rate in zip(crash_values, expected_rates):
        print(f"The quantity of days with returns smaller "
              f"than {crash_value * 100}% is equal to {len(r[r < crash_value])}, "
              f"empirically we observe in markets {crash_rate}.")

"""
Functions for sampling bubbles of 2 dimensions.
By setting the parameters of the second dimension to zero, we recover a type-I bubble.
"""
import math
import time
import typing

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.stock_bubble.delay_compens import Delay_compens
from src.stock_bubble.general_routines import X_next_by_induct, INVAR_next_induc, garch_model, compute_weight_Wt
from src.stock_bubble.store_argextr import Store_argExtr
from src.stock_bubble.updater_bubble_flags import Updater_bubble_flags
from src.stock_bubble.utility import logistic_function_for_X


def sample_self_exciting_bubble(nb_bins_total: int, initial_conditions, charac_time_a_taus: npt.ArrayLike,
                                weight_charac_time_a_taus: npt.ArrayLike, X_bar, r_bar, sigma_bar,
                                s_logistic_fct: npt.ArrayLike, exo_rate_nus: npt.ArrayLike,
                                exo_rate_nus_2: npt.ArrayLike, endo_rate_etas: npt.ArrayLike,
                                endo_rate_etas_2: npt.ArrayLike, jump_size_kappas: npt.ArrayLike, alpha_garch: float,
                                beta_garch: float, delta_INVAR, kernel, deltas_local_computing: typing.Iterable,
                                weights_deltas, burnin: bool, store_stop_time: bool = False,
                                use_delay_compensation: bool = True, remove_all_mispricing_mechanism: bool = False):
    """
    Sample a trajectory of a self-exciting bubble.

    Args:
        nb_bins_total: total number of bins.
        initial_conditions: initial conditions for the simulation constituted by X0, sigma0*sigma0, S0, r0. X0 is a single value irrespective whether there are multiple mispricing indices.
        charac_time_a_taus: the parameter a_tau as in the paper, where multiple values can be given for different scales.
        weight_charac_time_a_taus: the weights of the different values of a_tau used in the summation.
        X_bar: Mean value for the X (mispricing) process.
        r_bar: Mean value for the r (interest rate) process.
        sigma_bar: Mean value for the sigma (volatility) process.
        s_logistic_fct: the parameter s in the logistic function for X.
        exo_rate_nus: exogenous rate. Shape (2,).
        exo_rate_nus_2: exogenous rate scaling the value L(X_t). Shape (2,).
        endo_rate_etas: endogenous rate. Shape (2, 2).
        endo_rate_etas_2: endogenous rate scaling the value L(X_t). Shape (2, 2).
        jump_size_kappas: jump size. Shape (2,).
        alpha_garch: parameter for GARCH model.
        beta_garch: parameter for GARCH model.
        delta_INVAR: the time step for the INVAR process.
        kernel:
        deltas_local_computing:
        weights_deltas:
        burnin: if True, we add a burn in period to the simulation.
        store_stop_time: if True, we store the argmin and argmax for the last time step.
                        Used for the experiments in `experiments/delayed_compensation/show_evolution_stopping_time.py`.
        use_delay_compensation: activates the delay compensation mechanism.
        remove_all_mispricing_mechanism: not technically necessary, but useful to quickly remove all mispricing; it sets the value of L(X_t) to 0 for all times.

    Returns:

    """
    # the parameter store_stop_time is for some plots, we want to see exactly how the arg extremums behave.
    # WIP not sure the two are necessary
    assert (burnin != store_stop_time) or \
           not (burnin and store_stop_time), \
        f"If burn_in is True, then store_stop_time is False, and reversely. Here bu" \
        f"rn_in is {burnin} and store_stop_time {store_stop_time}."

    if burnin:
        NB_ADDIT_STEP_BC_BURNIN = 250
        nb_bins_total += delta_INVAR * NB_ADDIT_STEP_BC_BURNIN

    time.sleep(0.1)  # we wait so everything is printed correctly

    ### containers
    S = np.zeros(nb_bins_total + 1)
    S_bm = np.zeros(nb_bins_total + 1)
    S_up_jumps = np.zeros(nb_bins_total + 1)
    S_down_jumps = np.zeros(nb_bins_total + 1)

    r = np.zeros(nb_bins_total + 1)
    r_bm = np.zeros(nb_bins_total + 1)
    r_up_jumps = np.zeros(nb_bins_total + 1)
    r_down_jumps = np.zeros(nb_bins_total + 1)

    sigma2 = np.zeros(nb_bins_total + 1)
    X = np.zeros(
        (len(charac_time_a_taus), nb_bins_total + 1))  # matrix, each line represents one time of the mispricing index
    N = np.zeros((2, nb_bins_total + 1))
    lambda_t = np.zeros((2, nb_bins_total + 1))

    epsilon = np.random.normal(size=nb_bins_total + 1)
    mispricing_intensity = np.zeros(nb_bins_total + 1)

    ### initial condition
    _, sigma2[0], S[0], r[0] = initial_conditions
    X[:, 0] += initial_conditions[
        0]  # we add the initial condition to the mispricing index in such a way because X could be multivariate

    S_bm[0] = S_up_jumps[0] = S_down_jumps[0] = math.pow(S[0], 1. / 3.)  # we equally distribute

    all_indc = np.arange(start=nb_bins_total, stop=0, step=-1)

    # list_flagDD represents whether each time-scale indicates a drawdown or not.
    list_flagDD = [False] * len(deltas_local_computing)
    # When flags changed, so we can keep track of the regimes.
    list_regime_change = [[0] + list_flagDD.copy()]  # at time zero, you start with regime 0

    # temporary variables for each time step.
    nu = np.zeros_like(exo_rate_nus)
    eta = np.zeros_like(endo_rate_etas)

    if use_delay_compensation:
        ### compensation
        # uncompensated_part = Delay_compens(0.02, "uniform")         # around 10 days of compensation
        uncompensated_part = Delay_compens(0.018, "normal")  # around 10 days of compensation
        # no compensation
        # uncompensated_part = Delay_compens(0.0, "None")

    ### storing the arg extremums
    # we still create it bc we need to hand back the elements at the end of
    # the simulation despite having store_stop_time is false. It is up to the user to not use it outside
    if store_stop_time:
        argExt = Store_argExtr()
    else:
        argExt = None

    updater_bubble_flags = Updater_bubble_flags(deltas_local_computing)

    ### Evolution Dynamics
    for t_index in tqdm(range(1, nb_bins_total + 1)):
        X[:, t_index] = X_next_by_induct(charac_time_a_taus, X[:, t_index - 1], X_bar,
                                         s_logistic_fct, r[t_index - 1], r_bar,
                                         remove_all_mispricing_mechanism)

        mispricing_intensity[t_index] = logistic_function_for_X(X[:, t_index], weight_charac_time_a_taus)

        nu[0] = exo_rate_nus[0] + exo_rate_nus_2[0] * mispricing_intensity[t_index]
        nu[1] = exo_rate_nus[1] + exo_rate_nus_2[1] * mispricing_intensity[t_index]

        eta[0, 0] = endo_rate_etas[0, 0] + endo_rate_etas_2[0, 0] * mispricing_intensity[t_index]
        eta[1, 1] = endo_rate_etas[1, 1] + endo_rate_etas_2[1, 1] * mispricing_intensity[t_index]

        lambda_t[:, t_index] = INVAR_next_induc(nu, eta, N[:, 1:t_index], kernel,
                                                all_indc[nb_bins_total - t_index + 1:])

        N[:, t_index] = np.random.poisson(lambda_t[:, t_index])

        # GARCH model
        sigma2[t_index] = garch_model(alpha_garch, beta_garch, alpha_factor=r[t_index - 1] - r_bar,
                                      beta_factor=sigma2[t_index - 1],
                                      rest_factor=sigma_bar * sigma_bar)

        # update r[t_index] as a type-II bubble
        (r_bm[t_index], r_up_jumps[t_index],
         r_down_jumps[t_index]) = next_day_returns(r_bar, sigma2[t_index], epsilon[t_index], jump_size_kappas,
                                                   N[:, t_index], delta_INVAR, lambda_t[:, t_index], list_flagDD,
                                                   weights_deltas)
        r[t_index] = r_bm[t_index] + r_up_jumps[t_index] + r_down_jumps[t_index]

        if use_delay_compensation:
            delay_comp = uncompensated_part.delayed_compensation(delta_INVAR, list_flagDD, jump_size_kappas,
                                                                 lambda_t, t_index)
            r_down_jumps[t_index] -= delay_comp
            r[t_index] -= delay_comp

        # back to "normal" price (compared to log normal price)
        S[t_index] = S[t_index - 1] * np.exp(r[t_index])
        S_bm[t_index] = S_bm[t_index - 1] * np.exp(r_bm[t_index])
        S_up_jumps[t_index] = S_up_jumps[t_index - 1] * np.exp(r_up_jumps[t_index])
        S_down_jumps[t_index] = S_down_jumps[t_index - 1] * np.exp(r_down_jumps[t_index])

        # change list_flagDD state if needed:
        (argmin_current, argmax_current,
         index_time_minus_delta) = updater_bubble_flags.update_flagBubble(S, list_flagDD, list_regime_change, t_index)

        ### Stores the extremums
        if store_stop_time:
            # store the argmin and argmax in a list
            argExt.add_extr(argmin_current, argmax_current, index_time_minus_delta, t_index)

    # converts the vector of flag whether is in bubble mode or not.
    list_regime_change = np.array(list_regime_change,
                                  dtype=int).transpose()  # line one gives time, line two gives regimes

    if store_stop_time:
        (argmax_list, argmin_list, argmax_corresponding_times,
         argmin_corresponding_times) = argExt.return_res()
    else:
        (argmax_list, argmin_list, argmax_corresponding_times,
         argmin_corresponding_times) = (None, None, None, None)

    ### finally we prepare the output variables
    if burnin:  # remove the burning-in period
        ret = [S[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               S_bm[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:], S_up_jumps[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               S_down_jumps[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               r[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               r_bm[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:], r_up_jumps[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               r_down_jumps[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               sigma2[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               X[:, delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               N[:, delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               mispricing_intensity[delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               lambda_t[:, delta_INVAR * NB_ADDIT_STEP_BC_BURNIN:],
               list_regime_change,  # we deal with it later, position 7
               argmin_list, argmax_list,  # we do not change these output depending on burn-in.
               argmin_corresponding_times,
               argmax_corresponding_times]

        # list_regime_change has to be treated differently, we slice and shift the indices back.
        nb_of_phases_during_burn_in = np.argmax(delta_INVAR * NB_ADDIT_STEP_BC_BURNIN < list_regime_change[0, :])
        if nb_of_phases_during_burn_in == 0:  # means all phases are during burn_in since there is no phase OUTSIDE the burn-in period.
            nb_of_phases_during_burn_in = len(list_regime_change[0, :])

        # slicing the elements
        list_regime_change_with_burn_in_period = \
            list_regime_change[:, nb_of_phases_during_burn_in - 1:]  # -1 bc we still want the first one
        list_regime_change_with_burn_in_period[0] -= \
            delta_INVAR * NB_ADDIT_STEP_BC_BURNIN  # might be empty. If array is empty, then stays empty
        list_regime_change_with_burn_in_period[
            0, 0] = 0  # we change the first value, which is not correctly indexed in time.

        list_regime_change = list_regime_change_with_burn_in_period
        # list_regime_change = np.insert(list_regime_change_with_burn_in_period, 0, 0, axis=1)  # a dictionary would have been better
        ret[13] = list_regime_change

    else:
        ret = [S, S_bm, S_up_jumps, S_down_jumps,
               r, r_bm, r_up_jumps, r_down_jumps,
               sigma2, X, N, mispricing_intensity,
               lambda_t, list_regime_change,
               argmin_list, argmax_list,
               argmin_corresponding_times,
               argmax_corresponding_times]

    return tuple(ret)


def next_day_returns(mu_t, sigma2_t, epsilon_t, kappa, N_t, delta_INVAR, lambda_t, list_flagDD, weights_deltas):
    """
    mu_t: returns per day
    sigma2_t: vol per day
    """
    r_bm = mu_t + math.sqrt(sigma2_t) * epsilon_t

    weight_Wt = compute_weight_Wt(list_flagDD, weights_deltas)

    r_up_jumps = kappa[0] * (N_t[0] - weight_Wt * delta_INVAR * lambda_t[0])
    r_down_jumps = kappa[1] * (N_t[1] - delta_INVAR * lambda_t[1])
    return r_bm, r_up_jumps, r_down_jumps

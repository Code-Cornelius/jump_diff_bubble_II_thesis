import numpy as np
import numpy.typing as npt

def garch_model(alpha, beta, alpha_factor, beta_factor, rest_factor):
    return alpha * alpha_factor * alpha_factor + \
           beta * beta_factor + \
           (1 - alpha - beta) * rest_factor


def X_next_by_induct(the_as_delta: npt.ArrayLike, Xtm1: npt.ArrayLike, X_bar: float, s_logistic_fct: npt.ArrayLike,
                     rtm1: float, r_bar: float, remove_all_mispricing_mechanism: bool):
    if remove_all_mispricing_mechanism:
        return np.full_like(Xtm1, -1E3)

    return the_as_delta * Xtm1 + (1. - the_as_delta) * X_bar + (1. - the_as_delta) / s_logistic_fct * (rtm1 - r_bar)


def INVAR_next_induc(mu, branching_ratio, past_events, kernel, indices4kernel):
    # works even when t = 1

    # we are first applying the multiplicative factors gemv way eta * J, for each time 1:t.
    # Then, we are applying our filter / kernel on it.
    # It has to be repeated so the kernel is 2-D.
    # the operation on kernel_int_geom: add dimension and then repeat.

    # we can slice past_events and indices4kernel for increase speed
    past_events_loc = past_events[:, -10000:]  # correspond to a 40 years period of influence.
    indices4kernel_loc = indices4kernel[-10000:]  # after  some testing, the influence is minimal above 40 y.
    return mu + np.sum(branching_ratio @ past_events_loc
                       * kernel(indices4kernel_loc)[None, :].repeat(2, axis=0), axis=1)


def compute_weight_Wt(list_flagDD, weights_deltas):
    if not len(list_flagDD):  # if empty
        return 0
    weights_filtered = [flag * weight for flag, weight in zip(list_flagDD, weights_deltas)]
    weight_Wt = sum(weights_filtered)
    return weight_Wt

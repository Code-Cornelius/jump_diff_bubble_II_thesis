import math

import numpy as np


class Delay_compens():
    # we chose to use a class because that way we can monitor
    # the delayed compensation (what is left to release)
    # but also how much there was at the beginning of the drawdown.

    def __init__(self, scale, type):
        self.delayed_compensation_left = 0.
        self.delayed_compensation_total = 0.  # we store these two variables.
        # the first one delayed_compensation_left corresponds to how much is left to release.
        # whereas delayed_compensation_total is the value at start.
        # It is a good factor so the number of days for releasing the delayed compensation is tractable.

        self.scale = scale
        if type == "uniform":
            self.gen = lambda: np.random.uniform(0, scale, size=1)
        elif type == "normal":
            self.gen = lambda: np.abs(np.random.normal(size=1, scale=math.sqrt(scale)))
        elif type == "None":
            self.gen = lambda: 0.
        else:
            raise ValueError("Type is either 'uniform' or 'normal' or 'None'. You gave: ", type)

    def delayed_compensation(self, delta_INVAR, list_flagDD, kappa_val_jump_matrix, lambda_t, t):
        # Storing the missed compensation from bubble phases.

        # we compute f_i outside the branch such that we can monitor better changes and not see randomness
        f_i = self.gen()  # compute a random fraction to return
        p_i = np.random.uniform(0, 1.0, size=1)

        weight_Wt = sum(list_flagDD) / len(list_flagDD)
        # weight 0. means no compensation reintroduced, so we need to compensate 1. - weight.

        # all case scenario we accumulate more compensation, with value corresponding to which regime we are in:
        self.delayed_compensation_left += (1. - weight_Wt) * kappa_val_jump_matrix[0] * delta_INVAR * lambda_t[:, t][0]

        ################# this is alternative
        # self.delayed_compensation_total = self.delayed_compensation_left  # and the total is what is stored.
        ################# this is alternative

        if weight_Wt == 0.:
            self.delayed_compensation_total = self.delayed_compensation_left  # and the total is what is stored.

        ### now we might or not release some:
        # we sample p_i ~ uniform [0,1]. If it is higher than the weight (representing a probability), then we do nothing.
        # Hence in cases where we do not compensate (weight small), we will not return much either!
        # it is a rejection method; we do not return any delayed compensation

        # if p_i > weight_Wt:
        #     do not return delayed comp, so do nothing
        # return 0.
        ################# this is alternative
        if weight_Wt < 1.:
            return 0.
        ################# this is alternative

        else:  # we release the compensation
            release = self.delayed_compensation_total * f_i

            if self.delayed_compensation_left < release:  # case we just emptied the delayed compensation.
                release = self.delayed_compensation_left
                self.delayed_compensation_left = 0

            else:  # still delayed_compensation_left not 0.
                self.delayed_compensation_left -= release

            return release

class Store_argExtr(object):
    """
    This class is used to store the argmin and argmax of the stock price.
    Mostly written as a helper for the script `experiments/delayed_compensation/show_evolution_stopping_time.py`.
    """

    def __init__(self):

        self.argmax = [0]
        self.argmin = [0]
        # we store in this list the indices of the argmin and argmax
        # hence, the first min and max is necessarily the first point.

        # lists of lists
        self.argmax_corresponding_times = [[0]]
        self.argmin_corresponding_times = [[0]]

    def add_extr(self, argmin, argmax, index_time_minus_delta, current_t):

        # case we updated the argmin
        if self.argmin[-1] != argmin + index_time_minus_delta:
            self.argmin.append(argmin + index_time_minus_delta)
            self.argmin_corresponding_times.append([current_t])

        # otherwise add the current time to the list
        else:
            self.argmin_corresponding_times[-1].append(current_t)

        # case we updated the argmax
        if self.argmax[-1] != argmax + index_time_minus_delta:
            self.argmax.append(argmax + index_time_minus_delta)
            self.argmax_corresponding_times.append([current_t])

        # otherwise add the current time to the list
        else:
            self.argmax_corresponding_times[-1].append(current_t)

    def return_res(self):
        return (self.argmax, self.argmin,
                self.argmax_corresponding_times, self.argmin_corresponding_times)

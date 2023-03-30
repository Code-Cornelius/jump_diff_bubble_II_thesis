"""
Signal processing for a BM.
"""

import numpy as np
import scipy.signal as sig
from corai_plot import APlot

# section ######################################################################
#  #############################################################################
#  signal
sampling_frequency = int(1e5)
print("Sampling Frequency, in measurements per seconds:", sampling_frequency)
tt = np.linspace(0, 1, sampling_frequency)  # always the same length, regardless of nb points.

######################
## Two types of signal:
# signal = np.cumsum(np.random.normal(0, 2, sampling_frequency))
signal = np.random.normal(0, 0.15 / np.sqrt(250), sampling_frequency)
######################

# section ######################################################################
#  #############################################################################
#  PSD periodogram
# compute power spectral density
sampled_frequencies_period, power_sde_period = sig.periodogram(signal, sampling_frequency, scaling="density",
                                                               return_onesided=False, window='hamming')

freqs = np.fft.fftfreq(tt.size, 1 / sampling_frequency)
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(signal)) / sampling_frequency  # rescale to get right norm
ps *= ps

sampled_frequencies_fft = freqs[idx]
power_sde_fft = ps[idx]

variance_period = np.sum(power_sde_period)
print("Total Variance, for periodogram: ", variance_period)
variance_fft = np.sum(ps)
print("Total Variance, for fft :        ", variance_fft)

######## compute the periodogram and the auto-covariance sequence.
beg = len(signal) * 10 // 50
end = len(signal) * 40 // 50
slicer = slice(beg, end)  # we take out of the whole length and focus on the middle part (the 24 and 26th /50 segment)
auto_correl = np.correlate(signal[slicer], signal[slicer], mode='full')
print('Autocorrelation sequence and length: ', auto_correl, len(auto_correl))

# section ######################################################################
#  #############################################################################
#  plot
a = APlot(how=(3, 1))
a.uni_plot(nb_ax=0, xx=tt, yy=signal,
           dict_plot_param={'markersize': 0,
                            'color': 'r'},
           dict_ax={'ylabel': 'Original Signal',
                    'title': '',
                    'xlabel': 'Time [seconds]'})

a.uni_plot(nb_ax=1, xx=sampled_frequencies_fft, yy=power_sde_fft,
           dict_plot_param={'markersize': 0, 'linewidth': 1,
                            'color': 'g',
                            'label': 'fft'},
           dict_ax={'yscale': 'log',
                    'ylabel': 'Power spectral density [V**2/Hz]',
                    'title': '',
                    'xlabel': 'frequency [Hz]',
                    'ylim': [1e-10, 1e4]})
a._axs[-1].acorr(auto_correl, maxlags=len(auto_correl) // 2, linewidth=3.5)
a.set_dict_ax(nb_ax=2, dict_ax={'ylabel': 'Autocorrelation Sequence', 'title': '', 'xlabel': 'Lag $n$'})
a.show_legend(1)
a.tight_layout()

# adding the expected Autocorrelation
help_fct = lambda x: 1. / 2. * (np.abs(x + 1) - 2. * np.abs(x) + np.abs(x - 1))
a.uni_plot(nb_ax=2,
           xx=np.arange(-(end - beg), (end - beg), 1),
           yy=help_fct(np.arange(-(end - beg), (end - beg), 1)),
           dict_plot_param={'color': 'r', 'linewidth': 2., 'label': 'Expected Autocorrelation'})
a.show_plot()

"""
This script is a collection of signal processing experiments, where we show
the impact of trends, periodic signals and noise on the ACF and power spectral density.
"""
import numpy as np
import scipy.signal as sig
from corai_plot import APlot

# section ######################################################################
#  #############################################################################
#  signal
sampling_frequency = int(5e3)
print("Sampling Frequency, in measurements per seconds:", sampling_frequency)
frequency_base = 100  # frequency of the sinus
tt = np.linspace(0, 1, sampling_frequency)  # always the same length, regardless of nb points.
amp_base = 3.  # amplitude of the sinus
noise_power = 10. / sampling_frequency  # sigma_sqr * delta
signal = amp_base * np.sin(2 * np.pi * frequency_base * tt)
signal += np.random.normal(scale=np.sqrt(noise_power), size=(sampling_frequency,))  # corruption
######################################################## add a trend by uncommenting
# signal += 2 * amp_base * tt / tt[-1]
########################################################

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

# section ######################################################################
#  #############################################################################
#  autocorrelation sequence
######## compute the periodogram and the auto-covariance sequence.
slicer = slice(len(signal) * 24 // 50, len(signal) * 26 // 50)
auto_correl = np.correlate(signal[slicer], signal[slicer], mode='full')
print('Auto correlation sequence and length: ', auto_correl, len(auto_correl))

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
a.uni_plot(nb_ax=1, xx=sampled_frequencies_period, yy=power_sde_period,
           dict_plot_param={'markersize': 0, 'linewidth': 1,
                            'color': 'b',
                            'label': 'periodogram/welch'},
           dict_ax={'yscale': 'log',
                    'ylabel': 'power spectral density [V**2/Hz]',
                    'title': '',
                    'xlabel': 'frequency [Hz]',
                    'ylim': [1e-10, 1e4]})
a.uni_plot(nb_ax=1, xx=sampled_frequencies_fft, yy=power_sde_fft,
           dict_plot_param={'markersize': 0, 'linewidth': 1,
                            'color': 'g',
                            'label': 'fft'},
           dict_ax={'yscale': 'log',
                    'ylabel': 'Power spectral density [V**2/Hz]',
                    'title': '',
                    'xlabel': 'frequency [Hz]',
                    'ylim': [1e-10, 1e4]})
a._axs[-1].acorr(auto_correl, maxlags=len(auto_correl) // 2)
a.set_dict_ax(nb_ax=2, dict_ax={'ylabel': 'Autocorrelation Sequence', 'title': '', 'xlabel': 'Lag $n$'})
a.show_legend(1)
a.tight_layout()
a.show_plot()

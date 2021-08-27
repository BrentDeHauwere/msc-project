# %%

import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import window

OUTPUT_DIR = 'data/plots/'

# no. oscillations (cycles) that occur each second of time
ORDINARY_FREQUENCY = 4

SAMPLING_FREQUENCY = 55

# points to deduct for the incomplete sine wave
INCOMPLETE_MINUS_POINTS = 6


def plot_data(x_data, y_data, x_label, y_label, fig_name, title, y_lim=None, plot_type='plot'):
    fig = plt.figure()

    with sns.axes_style('whitegrid'):
        ax = fig.add_subplot()

    ax.set_xlim(x_data[0], x_data[-1])

    if y_lim != None:
        ax.set_ylim(y_lim[0], y_lim[1])

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # ax.set_title(title)

    if plot_type == 'plot':
        ax.plot(x_data, y_data)
    elif plot_type == 'stem':
        ax.stem(x_data, y_data)

    fig.savefig(fig_name)
    plt.show()
    plt.close(fig)


def plot_sine_wave_and_fft(slice_n=0):
    # create time axis
    time = np.arange(0, 1, 0.015)

    # create amplitude axis, based on sine wave
    amplitude = np.sin(2 * np.pi * ORDINARY_FREQUENCY * (time/time[-1]))

    # slice slice_n from waveform
    if slice_n != 0:
        time = time[:-slice_n]
        amplitude = amplitude[:-slice_n]

    # plot sine wave
    plot_data(time, amplitude, 'time (s)', 'amplitude',
              f'{OUTPUT_DIR}sin_wave.png' if slice_n == 0 else f'{OUTPUT_DIR}sin_wave-incomplete.png',
              title='4 Hz Sine Wave' if slice_n == 0 else 'Incomplete 4 Hz Sine Wave',
              plot_type='stem')

    # apply fft on sine wave
    fftdata = np.fft.fft(amplitude)
    fftdatafreq = np.empty_like(amplitude)
    for i in range(len(fftdata)):
        fftdatafreq[i] = abs(fftdata[i].real)

    # plot spectrum
    plot_data(np.arange(0, len(time) // 2), fftdatafreq[:len(time) // 2],
              'frequency (Hz)', 'amplitude',
              f'{OUTPUT_DIR}sin_wave_fft.png' if slice_n == 0 else f'{OUTPUT_DIR}sin_wave-incomplete_fft.png',
              title='FFT of 4 Hz Sine Wave' if slice_n == 0 else 'FFT of Incomplete 4 Hz Sine Wave',
              y_lim=[0, 20],
              plot_type='stem')


# COMPLETE SINE WAVE
plot_sine_wave_and_fft(slice_n=0)

# INCOMPLETE SINE WAVE
plot_sine_wave_and_fft(slice_n=INCOMPLETE_MINUS_POINTS)

# create time and amplitude axis, and slide
time = np.arange(0, 1, 1/SAMPLING_FREQUENCY)
amplitude = np.sin(2 * np.pi * ORDINARY_FREQUENCY *
                   (time/time[-1]))
time = time[:-INCOMPLETE_MINUS_POINTS]
amplitude = amplitude[:-INCOMPLETE_MINUS_POINTS]


# APPLY VARIOUS WINDOWS ON SIGNAL
for window_type in ['hann',
                    'hamm',
                    # 'blackman',
                    # 'boxcar',
                    # 'triang',
                    # 'bartlett',
                    'flattop',
                    # 'parzen',
                    # 'bohman',
                    # 'blackmanharris',
                    # 'nuttall',
                    # 'barthann',
                    # 'cosine',
                    # 'tukey',
                    # 'taylor'
                    ]:

    window_func = window(window_type, len(amplitude))
    plot_data(time, window_func, 'time (s)', 'amplitude', f'{OUTPUT_DIR}{window_type}.png',
              title=f'{window_type.title()} Window', plot_type='plot')

    # WINDOW IN FREQUENCY DOMAIN
    fftdata = np.fft.fft(window_func, 2048) / 25.5
    # shift the zero-frequency component to the center of the spectrum
    fftdata_mag = np.abs(np.fft.fftshift(fftdata))
    fftdatafreq = np.linspace(-0.5, 0.5, len(fftdata))
    with np.errstate(divide='ignore', invalid='ignore'):
        response = 20 * np.log10(fftdata_mag)
    response = np.clip(response, -100, 100)
    plot_data(fftdatafreq, response, 'normalized frequency (cycles per sample)', 'magnitude (dB)', f'{OUTPUT_DIR}{window_type}_frequency.png',
              title=f'Frequency Response of the {window_type.title()} Window', plot_type='plot')
    # plt.axis('tight')

    # plt.figure()
    # A = fft(window, 2048) / 25.5
    # mag = np.abs(fftshift(A))
    # freq = np.linspace(-0.5, 0.5, len(A))
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     response = 20 * np.log10(mag)
    # response = np.clip(response, -100, 100)

    # plt.plot(freq, response)
    # plt.title("Frequency response of the Hann window")
    # plt.ylabel("Magnitude [dB]")
    # plt.xlabel("Normalized frequency [cycles per sample]")
    # plt.axis('tight')
    # plt.show()

    amplitude_multiplied_window = amplitude * window_func
    plot_data(time, amplitude_multiplied_window, 'time (s)', 'amplitude', f'{OUTPUT_DIR}{window_type}_multiplied.png',
              title=f'{window_type.title()} Window', plot_type='stem')

    fftdata = np.fft.fft(amplitude_multiplied_window)
    fftdatafreq = np.zeros(len(amplitude))
    for i in range(len(fftdata)):
        fftdatafreq[i] = abs(fftdata[i].real)
    plot_data(np.arange(0, len(time) // 2), fftdatafreq[:len(time) // 2], 'frequency (Hz)',
              'amplitude', f'{OUTPUT_DIR}{window_type}_multiplied_fft.png', title=f'FFT of {window_type.title()} Windowed Incomplete 4 Hz Sine Wave', y_lim=[0, 20], plot_type='stem')

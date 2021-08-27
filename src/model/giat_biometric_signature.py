# # %%
# from scipy.fftpack import fft
# from skimage.filters import window
# from skimage.data import astronaut
# from skimage.color import rgb2gray
# from skimage import img_as_float
# from scipy.fftpack import fft2, fftshift
# import numpy as np
# import matplotlib.pyplot as plt
# from numpy.fft import fft, fftshift

# # number of points in the output window
# M = 51

# window = np.hanning(M)

# # plot hanning window
# plt.plot(window)
# plt.title("Hann window")
# plt.ylabel("Amplitude")
# plt.xlabel("Sample")
# plt.show()

# # plot frequency response
# plt.figure()
# A = fft(window, 2048) / 25.5
# mag = np.abs(fftshift(A))
# freq = np.linspace(-0.5, 0.5, len(A))
# response = 20 * np.log10(mag)
# response = np.clip(response, -100, 100)
# plt.plot(freq, response)
# plt.title("Frequency response of the Hann window")
# plt.ylabel("Magnitude [dB]")
# plt.xlabel("Normalized frequency [cycles per sample]")
# plt.axis('tight')
# plt.show()

# # %%
# # Fast Fourier transforms (FFTs) assume that the data being transformed represent one period of a periodic signal.
# # Thus the endpoints of the signal to be transformed can behave as discontinuities in the context of the FFT.
# # These discontinuities distort the output of the FFT, resulting in energy from “real” frequency components leaking into wider frequencies.

# # The effects of spectral leakage can be reduced by multiplying the signal with a window function.
# # Windowing smoothly reduces the amplitude of the signal as it reaches the edges,
# # removing the effect of the artificial discontinuity that results from the FFT.

# # In this example, we see that the FFT of a typical image can show strong spectral leakage along the x and y axes (see the vertical and horizontal lines in the figure).
# # The application of a two-dimensional Hann window greatly reduces the spectral leakage, making the “real” frequency information more visible in the plot of the frequency component of the FFT.


# image = img_as_float(rgb2gray(astronaut()))

# wimage = image * window('hann', image.shape)

# image_f = np.abs(fftshift(fft2(image)))
# wimage_f = np.abs(fftshift(fft2(wimage)))

# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# ax = axes.ravel()
# ax[0].set_title("Original image")
# ax[0].imshow(image, cmap='gray')
# ax[1].set_title("Windowed image")
# ax[1].imshow(wimage, cmap='gray')
# ax[2].set_title("Original FFT (frequency)")
# ax[2].imshow(np.log(image_f), cmap='magma')
# ax[3].set_title("Window + FFT (frequency)")
# ax[3].imshow(np.log(wimage_f), cmap='magma')
# plt.show()

# # %%

# import matplotlib.pyplot as plt
# import numpy as np
# from skimage.filters import window
# from scipy.fftpack import fft

# import seaborn as sns

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import ListedColormap

# BODY_PART = 'thigh'

# INPUT_RAD_DIR = 'data/processed/rad_body-part/'
# OUTPUT_DIR = 'data/plots/'

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlabel('Component 1')
# ax.set_ylabel('Component 2')
# ax.set_zlabel('Component 3')

# # all sequences we want to analyse
# signals_all = {
#     13: ['008a013s00L', '008a013s01L', '008a013s02L', '008a013s03L'],
#     15: ['008a015s00L', '008a015s01L', '008a015s02L', '008a015s03L'],
#     16: ['008a016s00L', '008a016s01L', '008a016s02L', '008a016s03L'],
#     20: ['009a020s00L', '009a020s01L', '009a020s02L', '009a020s03L'],
#     21: ['010a021s00L', '010a021s01L', '010a021s02L', '010a021s03L'],
#     22: ['010a022s00L', '010a022s01L', '010a022s02L', '010a022s03L'],
# }

# for subject, samples in signals_all.items():

#     # retrieve all radial acceleration files of a subject
#     signals = [np.load(f'{INPUT_RAD_DIR}{seq_id}_rad_{BODY_PART}.npy') for seq_id in samples]

#     for signal in signals:

#         print('signal shape', signal.shape)

#         # apply hann window on signal
#         hann_window = window('hann', len(signal))
#         windowed_signal = signal * hann_window
#         print('windowed signal shape', windowed_signal.shape)

#         # compute the 1-D discrete Fourier Transform
#         windowed_signal_f = fft(windowed_signal)
#         print('fft', windowed_signal_f.shape)

#         # compute the magnitude and phase of the 3 first components
#         magnitude = np.abs(windowed_signal_f[0:3])
#         phase = np.angle(windowed_signal_f[0:3])
#         biometric_signature = magnitude * phase
#         print(magnitude, phase, biometric_signature)

#         ax.scatter(biometric_signature[0],
#                 biometric_signature[1],
#                 biometric_signature[2],
#                 marker='o',
#                 c=sns.color_palette("Set2", 4)[0],
#                 cmap=ListedColormap(sns.color_palette("husl", 256).as_hex())
#         )

# plt.savefig(f'{OUTPUT_DIR}fft_biometric_signatures_{BODY_PART}.png')
# plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import window
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

BODY_PART = 'thigh'
N_COMPONENTS = 3
N_COMPONENT_START = 1
N_TRIALS = 1

INPUT_RAD_DIR = 'data/processed/rad_body-part/'
OUTPUT_DIR = 'data/plots/'


def apply_pca(x, n=3):

    # standardizing the features
    x = StandardScaler().fit_transform(x)

    # dimensionality reduction with pca
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(x)

    return principal_components


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel(f'Component {N_COMPONENT_START}')
ax.set_ylabel(f'Component {N_COMPONENT_START+1}')
ax.set_zlabel(f'Component {N_COMPONENT_START+2}')
# ax.set_title('Biometric Signature from ' + BODY_PART.title())


# all sequences we want to analyse
signals_all = {
    13: ['008a013s00L', '008a013s01L', '008a013s02L', '008a013s03L'],
    15: ['008a015s00L', '008a015s01L', '008a015s02L', '008a015s03L'],
    # 16: ['008a016s00L', '008a016s01L', '008a016s02L', '008a016s03L'],
    20: ['009a020s00L', '009a020s01L', '009a020s02L', '009a020s03L'],
    # 21: ['010a021s00L', '010a021s01L', '010a021s02L', '010a021s03L'],
    22: ['010a022s00L', '010a022s01L', '010a022s02L', '010a022s03L'],
    23: ['010a023s00L', '010a023s01L', '010a023s02L', '010a023s03L'],
    24: ['010a024s00L', '010a024s01L', '010a024s02L', '010a024s03L'],
    25: ['011a025s00L', '011a025s01L', '011a025s02L', '011a025s03L'],
    # 26: ['011a026s00L', '011a026s01L', '011a026s02L', '011a026s03L'],
}

for i in range(N_TRIALS):

    # dataframe to store the biometric signatures
    biometric_signatures_all_df = pd.DataFrame(
        columns=[*np.arange(N_COMPONENTS), 'subject_id'])

    # iterate over subjects
    for subject, samples in signals_all.items():

        # retrieve all radial acceleration files of a subject
        signals = [np.load(f'{INPUT_RAD_DIR}{seq_id}_rad_{BODY_PART}.npy')
                   for seq_id in samples]

        # apply hann window on signal
        hann_windows = [window('hann', len(signal)) for signal in signals]
        windowed_signals = [signal * hann_window
                            for signal, hann_window in zip(signals, hann_windows)]

        # compute the 1-D discrete Fourier Transform
        windowed_signals_f = [fft(windowed_signal)
                              for windowed_signal in windowed_signals]

        # compute the magnitude and phase of the 3 first components
        # note: term 0, alias DC term, is the average of all samples, because sine wave values are 0
        n_component_start = N_COMPONENT_START + i
        magnitudes = [np.abs(windowed_signal_f[n_component_start:N_COMPONENTS+n_component_start])
                      for windowed_signal_f in windowed_signals_f]
        phases = [np.angle(windowed_signal_f[n_component_start:N_COMPONENTS+n_component_start])
                  for windowed_signal_f in windowed_signals_f]
        biometric_signatures = [
            magnitude * phase for magnitude, phase in zip(magnitudes, phases)]

        # biometric_signatures = apply_pca(biometric_signatures)

        # add biometric signatures of subject to summary
        biometric_signatures_df = pd.DataFrame(biometric_signatures)
        biometric_signatures_df['subject_id'] = subject
        biometric_signatures_all_df = pd.concat(
            [biometric_signatures_all_df, biometric_signatures_df])

        # plot samples of subject
        ax.scatter([biometric_signature[0]
                    for biometric_signature in biometric_signatures],
                   [biometric_signature[1]
                    for biometric_signature in biometric_signatures],
                   [biometric_signature[2]
                    for biometric_signature in biometric_signatures],
                   label=f'subject {subject}',
                   alpha=1.0)

    ax.legend(loc='upper right', prop={'size': 6})
    plt.savefig(f'{OUTPUT_DIR}fft_biometric_signatures_{BODY_PART}.png')
    plt.show()

    X = biometric_signatures_all_df.drop('subject_id', axis=1)
    y = biometric_signatures_all_df['subject_id']

    print(
        # values between -1 and +1
        'silhouette_score', silhouette_score(X, y),

        # higher relates to a model with better defined clusters
        'calinski_harabasz_score', calinski_harabasz_score(X, y),

        # zero is the lowest possible score, and values closer to zero indicate a better partition
        'davies_bouldin_score', davies_bouldin_score(X, y))

# trace_intra = []
# trace_inter = []

# for s in biometric_signatures_all_df.subject_id.unique():

#     # select subject samples and non-subject samples
#     set_s = biometric_signatures_all_df[biometric_signatures_all_df.subject_id == s].drop(
#         'subject_id', axis=1)
#     set_not_s = biometric_signatures_all_df[biometric_signatures_all_df.subject_id != s].drop(
#         'subject_id', axis=1)

#     # calculate the covariance matrix
#     # transpose as numpy interprets each row as a variable, and each column a single observation
#     cov_s = np.cov(set_s.transpose())
#     cov_not_s = np.cov(set_not_s.transpose())

#     # calculate trace of covariance matrix
#     trace_s = np.trace(cov_s)
#     trace_not_s = np.trace(cov_not_s)

#     # save trace
#     trace_intra.append(trace_s)
#     trace_inter.append(trace_not_s)


# fig_variance, ax_variance = plt.subplots()
# ax_variance.set_title(
#     'Intra and Inter-Variance of Biometric Signatures from ' + BODY_PART.title())
# ax_variance.boxplot([trace_intra, trace_inter],
#                     labels=['Intra-Variance', 'Inter-Variance'])
# plt.savefig(f'{OUTPUT_DIR}variance_biometric_signatures_{BODY_PART}.png')
# plt.show()

# %%

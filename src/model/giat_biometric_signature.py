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

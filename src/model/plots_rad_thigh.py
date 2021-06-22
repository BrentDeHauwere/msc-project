# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

OUTPUT_DIR = 'data/results/'

SIGMA = 1
TRUNCATE = 1  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size

rad_avg_arr2 = np.load(f'{OUTPUT_DIR}009a017s02L_rad_thigh.npy')
rad_avg_arr3 = np.load(f'{OUTPUT_DIR}009a017s03L_rad_thigh.npy')

rad_avg_arr2_gaus = gaussian_filter1d(
    rad_avg_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_avg_arr3_gaus = gaussian_filter1d(
    rad_avg_arr3, sigma=SIGMA, truncate=TRUNCATE)

for pname, rad2, rad3 in [['RAW', rad_avg_arr2, rad_avg_arr3],
                          ['GAUS', rad_avg_arr2_gaus, rad_avg_arr3_gaus]]:

    sns.scatterplot(x=range(len(rad2)),
                    y=rad2,
                    label='subject ' + '009a017s02L')
    sns.scatterplot(x=range(len(rad3)),
                    y=rad3,
                    label='subject ' + '009a017s03L')

    plt.title('Average Radial Acceleration in Thigh')
    plt.xlabel('x')
    plt.ylabel('avg radial acceleration (magnitude)')
    plt.legend(loc='upper right')
    plt.savefig(
        OUTPUT_DIR + 'plots/average_radial_acceleration_in_thigh_' + pname)
    plt.show()
    plt.clf()

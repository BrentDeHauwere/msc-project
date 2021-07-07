# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

OUTPUT_DIR = 'data/results/'

SIGMA = 1
TRUNCATE = 1  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size

# average radial acceleration in body parts
rad_thigh_arr2 = np.load(f'{OUTPUT_DIR}009a017s02L_rad_thigh.npy')
rad_thigh_arr3 = np.load(f'{OUTPUT_DIR}009a017s03L_rad_thigh.npy')

rad_leg_arr2 = np.load(f'{OUTPUT_DIR}009a017s02L_rad_leg.npy')
rad_leg_arr3 = np.load(f'{OUTPUT_DIR}009a017s03L_rad_leg.npy')

rad_torso_arr2 = np.load(f'{OUTPUT_DIR}009a017s02L_rad_torso.npy')
rad_torso_arr3 = np.load(f'{OUTPUT_DIR}009a017s03L_rad_torso.npy')

# apply gaussian averaging
rad_thigh_arr2_gaus = gaussian_filter1d(
    rad_thigh_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr3_gaus = gaussian_filter1d(
    rad_thigh_arr3, sigma=SIGMA, truncate=TRUNCATE)

rad_leg_arr2_gaus = gaussian_filter1d(
    rad_leg_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr3_gaus = gaussian_filter1d(
    rad_leg_arr3, sigma=SIGMA, truncate=TRUNCATE)

rad_torso_arr2_gaus = gaussian_filter1d(
    rad_torso_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr3_gaus = gaussian_filter1d(
    rad_torso_arr3, sigma=SIGMA, truncate=TRUNCATE)

for bpname, type, rad2, rad3 in [['thigh', 'raw', rad_thigh_arr2, rad_thigh_arr3],
                                 ['thigh', 'gaus', rad_thigh_arr2_gaus,
                                     rad_thigh_arr3_gaus],
                                 ['leg', 'raw', rad_leg_arr2, rad_leg_arr3],
                                 ['leg', 'gaus', rad_leg_arr2_gaus,
                                     rad_leg_arr3_gaus],
                                 ['torso', 'raw', rad_torso_arr2, rad_torso_arr3],
                                 ['torso', 'gaus', rad_torso_arr2_gaus, rad_torso_arr3_gaus]]:

    sns.scatterplot(x=range(len(rad2)),
                    y=rad2,
                    label='subject ' + '009a017s02L')
    sns.scatterplot(x=range(len(rad3)),
                    y=rad3,
                    label='subject ' + '009a017s03L')

    plt.title(
        f'Average Radial Acceleration in {bpname.capitalize()} ({type.capitalize()})')
    plt.xlabel('x')
    plt.ylabel('avg radial acceleration (magnitude)')
    plt.legend(loc='upper right')
    plt.savefig(
        OUTPUT_DIR + f'plots/average_radial_acceleration_in_{bpname}_{type.upper()}')
    plt.show()
    plt.clf()

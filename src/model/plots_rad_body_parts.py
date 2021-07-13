# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# %% SOTON
OUTPUT_DIR = 'data/results/'

SIGMA = 2.5
TRUNCATE = 2  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size
print(W)

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

# %% CASIA

OUTPUT_DIR = 'data/results/'

SIGMA = 1.6
TRUNCATE = 2  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size
print(W)

# average radial acceleration in body parts
rad_thigh_arr1 = np.load(f'{OUTPUT_DIR}fyc-00_3_rad_thigh.npy')
rad_thigh_arr2 = np.load(f'{OUTPUT_DIR}hy-00_1_rad_thigh.npy')
rad_thigh_arr3 = np.load(f'{OUTPUT_DIR}ljg-00_3_rad_thigh.npy')
rad_thigh_arr4 = np.load(f'{OUTPUT_DIR}ml-00_1_rad_thigh.npy')
rad_thigh_arr5 = np.load(f'{OUTPUT_DIR}syj-00_3_rad_thigh.npy')

rad_leg_arr1 = np.load(f'{OUTPUT_DIR}fyc-00_3_rad_leg.npy')
rad_leg_arr2 = np.load(f'{OUTPUT_DIR}hy-00_1_rad_leg.npy')
rad_leg_arr3 = np.load(f'{OUTPUT_DIR}ljg-00_3_rad_leg.npy')
rad_leg_arr4 = np.load(f'{OUTPUT_DIR}ml-00_1_rad_leg.npy')
rad_leg_arr5 = np.load(f'{OUTPUT_DIR}syj-00_3_rad_leg.npy')

rad_torso_arr1 = np.load(f'{OUTPUT_DIR}fyc-00_3_rad_torso.npy')
rad_torso_arr2 = np.load(f'{OUTPUT_DIR}hy-00_1_rad_torso.npy')
rad_torso_arr3 = np.load(f'{OUTPUT_DIR}ljg-00_3_rad_torso.npy')
rad_torso_arr4 = np.load(f'{OUTPUT_DIR}ml-00_1_rad_torso.npy')
rad_torso_arr5 = np.load(f'{OUTPUT_DIR}syj-00_3_rad_torso.npy')

# apply gaussian averaging
rad_thigh_arr1_gaus = gaussian_filter1d(
    rad_thigh_arr1, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr2_gaus = gaussian_filter1d(
    rad_thigh_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr3_gaus = gaussian_filter1d(
    rad_thigh_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr4_gaus = gaussian_filter1d(
    rad_thigh_arr4, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr5_gaus = gaussian_filter1d(
    rad_thigh_arr5, sigma=SIGMA, truncate=TRUNCATE)

rad_leg_arr1_gaus = gaussian_filter1d(
    rad_leg_arr1, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr2_gaus = gaussian_filter1d(
    rad_leg_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr3_gaus = gaussian_filter1d(
    rad_leg_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr4_gaus = gaussian_filter1d(
    rad_leg_arr4, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr5_gaus = gaussian_filter1d(
    rad_leg_arr5, sigma=SIGMA, truncate=TRUNCATE)

rad_torso_arr1_gaus = gaussian_filter1d(
    rad_torso_arr1, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr2_gaus = gaussian_filter1d(
    rad_torso_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr3_gaus = gaussian_filter1d(
    rad_torso_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr4_gaus = gaussian_filter1d(
    rad_torso_arr4, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr5_gaus = gaussian_filter1d(
    rad_torso_arr5, sigma=SIGMA, truncate=TRUNCATE)

for bpname, type, rad1, rad2, rad3, rad4, rad5 in [['thigh', 'raw', rad_thigh_arr1, rad_thigh_arr2, rad_thigh_arr3, rad_thigh_arr4, rad_thigh_arr5],
                                                   ['thigh', 'gaus', rad_thigh_arr1_gaus, rad_thigh_arr2_gaus,
                                                       rad_thigh_arr3_gaus, rad_thigh_arr4_gaus, rad_thigh_arr5_gaus],
                                                   ['leg', 'raw', rad_leg_arr1, rad_leg_arr2,
                                                       rad_leg_arr3, rad_leg_arr4, rad_leg_arr5],
                                                   ['leg', 'gaus', rad_leg_arr1_gaus, rad_leg_arr2_gaus,
                                                       rad_leg_arr3_gaus, rad_leg_arr4_gaus, rad_leg_arr5_gaus],
                                                   ['torso', 'raw', rad_torso_arr1, rad_torso_arr2,
                                                    rad_torso_arr3, rad_torso_arr4, rad_torso_arr5],
                                                   ['torso', 'gaus', rad_torso_arr1_gaus, rad_torso_arr2_gaus, rad_torso_arr3_gaus, rad_torso_arr4_gaus, rad_torso_arr5_gaus]]:

    # sns.scatterplot(x=range(len(rad1)),
    #                 y=rad1,
    #                 label='subject ' + '1')
    # sns.scatterplot(x=range(len(rad2)),
    #                 y=rad2,
    #                 label='subject ' + '2')
    # sns.scatterplot(x=range(len(rad3)),
    #                 y=rad3,
    #                 label='subject ' + '3')
    # sns.scatterplot(x=range(len(rad4)),
    #                 y=rad4,
    #                 label='subject ' + '4')
    sns.scatterplot(x=range(len(rad5)),
                    y=rad5,
                    label='subject ' + '5')

    plt.title(
        f'Average Radial Acceleration in {bpname.capitalize()} ({type.capitalize()})')
    plt.xlabel('x')
    plt.ylabel('avg radial acceleration (magnitude)')
    plt.legend(loc='upper right')
    plt.savefig(
        OUTPUT_DIR + f'plots/average_radial_acceleration_in_{bpname}_{type.upper()}_CASIA')
    plt.show()
    plt.clf()

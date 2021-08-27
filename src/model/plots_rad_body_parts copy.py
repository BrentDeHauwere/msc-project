# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# %% SOTON
INPUT_RAD_DIR = 'data/processed/rad_body-part/'
OUTPUT_DIR = 'data/plots/'

SUBJECTS = ['009a017s02L', '009a017s03L', '009a017s04L']
# SUBJECTS = ['ljg-00_3', 'hy-00_1', 'syj-00_3']
SIGMA = 2.2
TRUNCATE = 2.5  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size
print(W)

# average radial acceleration in body parts
rad_thigh_arr2 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[0]}_rad_thigh.npy')
rad_thigh_arr3 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[1]}_rad_thigh.npy')
rad_thigh_arr4 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[2]}_rad_thigh.npy')[16:]

rad_leg_arr2 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[0]}_rad_leg.npy')
rad_leg_arr3 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[1]}_rad_leg.npy')
rad_leg_arr4 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[2]}_rad_leg.npy')[16:]

rad_torso_arr2 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[0]}_rad_torso.npy')
rad_torso_arr3 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[1]}_rad_torso.npy')
rad_torso_arr4 = np.load(f'{INPUT_RAD_DIR}{SUBJECTS[2]}_rad_torso.npy')[16:]

# apply gaussian averaging
rad_thigh_arr2_gaus = gaussian_filter1d(
    rad_thigh_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr3_gaus = gaussian_filter1d(
    rad_thigh_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_thigh_arr4_gaus = gaussian_filter1d(
    rad_thigh_arr4, sigma=SIGMA, truncate=TRUNCATE)

rad_leg_arr2_gaus = gaussian_filter1d(
    rad_leg_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr3_gaus = gaussian_filter1d(
    rad_leg_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_leg_arr4_gaus = gaussian_filter1d(
    rad_leg_arr4, sigma=SIGMA, truncate=TRUNCATE)

rad_torso_arr2_gaus = gaussian_filter1d(
    rad_torso_arr2, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr3_gaus = gaussian_filter1d(
    rad_torso_arr3, sigma=SIGMA, truncate=TRUNCATE)
rad_torso_arr4_gaus = gaussian_filter1d(
    rad_torso_arr4, sigma=SIGMA, truncate=TRUNCATE)

for bpname, type, rad2, rad3, rad4 in [
    ['thigh', 'gaus', rad_thigh_arr2_gaus,
     rad_thigh_arr3_gaus, rad_thigh_arr4_gaus],

    ['leg', 'gaus', rad_leg_arr2_gaus,
     rad_leg_arr3_gaus, rad_leg_arr4_gaus],

        ['torso', 'gaus', rad_torso_arr2_gaus, rad_torso_arr3_gaus, rad_torso_arr4_gaus]]:

    avg_incl = np.load(
        f'/Users/BrentDeHauwere/Documents/Academic_Archive/MSc Artificial Intelligence/MSc Project/Implementation/msc-project/009a017s02L_left {bpname}_incl.npy')
    avg_incl = (avg_incl - avg_incl.min()) / \
        (avg_incl.max() - avg_incl.min()) + 0.3

    # xi−min(x)max(x)−min(x)
    sns.scatterplot(x=range(len(rad2)),
                    y=(rad2 - rad2.min())/(rad2.max() - rad2.min())*2-1,
                    label='subject ' + SUBJECTS[0])
    sns.scatterplot(x=range(len(rad3)),
                    y=(rad3 - rad3.min())/(rad3.max() - rad3.min())*2-1,
                    label='subject ' + SUBJECTS[1])
    sns.scatterplot(x=range(len(rad4)),
                    y=(rad4 - rad4.min())/(rad4.max() - rad4.min())*2-1,
                    label='subject ' + SUBJECTS[2])
    sns.scatterplot(x=range(len(avg_incl)),
                    y=(avg_incl - avg_incl.min()) /
                    (avg_incl.max() - avg_incl.min())*2-1,
                    label='average inclination')

    # plt.title(
    #     f'Average Radial Acceleration in {bpname.capitalize()} ({type.capitalize()})')
    plt.xlabel('x')
    plt.ylabel('avg radial acceleration (magnitude)')
    plt.legend(loc='upper right')
    plt.savefig(
        OUTPUT_DIR + f'average_radial_acceleration_in_{bpname}_{type.upper()}_W{W}_S{SIGMA}.png')
    plt.show()
    plt.clf()

# %% CASIA

SIGMA = 1.6
TRUNCATE = 2  # determines the window size
W = 2*int(TRUNCATE*SIGMA + 0.5) + 1  # formula for window size
print(W)

# average radial acceleration in body parts
rad_thigh_arr1 = np.load(f'{INPUT_RAD_DIR}fyc-00_3_rad_thigh.npy')
rad_thigh_arr2 = np.load(f'{INPUT_RAD_DIR}hy-00_1_rad_thigh.npy')
rad_thigh_arr3 = np.load(f'{INPUT_RAD_DIR}ljg-00_3_rad_thigh.npy')
rad_thigh_arr4 = np.load(f'{INPUT_RAD_DIR}ml-00_1_rad_thigh.npy')
rad_thigh_arr5 = np.load(f'{INPUT_RAD_DIR}syj-00_3_rad_thigh.npy')

rad_leg_arr1 = np.load(f'{INPUT_RAD_DIR}fyc-00_3_rad_leg.npy')
rad_leg_arr2 = np.load(f'{INPUT_RAD_DIR}hy-00_1_rad_leg.npy')
rad_leg_arr3 = np.load(f'{INPUT_RAD_DIR}ljg-00_3_rad_leg.npy')
rad_leg_arr4 = np.load(f'{INPUT_RAD_DIR}ml-00_1_rad_leg.npy')
rad_leg_arr5 = np.load(f'{INPUT_RAD_DIR}syj-00_3_rad_leg.npy')

rad_torso_arr1 = np.load(f'{INPUT_RAD_DIR}fyc-00_3_rad_torso.npy')
rad_torso_arr2 = np.load(f'{INPUT_RAD_DIR}hy-00_1_rad_torso.npy')
rad_torso_arr3 = np.load(f'{INPUT_RAD_DIR}ljg-00_3_rad_torso.npy')
rad_torso_arr4 = np.load(f'{INPUT_RAD_DIR}ml-00_1_rad_torso.npy')
rad_torso_arr5 = np.load(f'{INPUT_RAD_DIR}syj-00_3_rad_torso.npy')

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
        OUTPUT_DIR + f'average_radial_acceleration_in_{bpname}_{type.upper()}_CASIA')
    plt.show()
    plt.clf()

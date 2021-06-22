# %%
import math
import os
import glob
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

INPUT_SEQUENCES_DIR = 'data/processed/sequences/'
INPUT_POSE_ESTIMATION_DIR = 'data/processed/pose_estimation/'
OUTPUT_DIR = 'data/results/'

N_POINTS = 25
POINT_LTHIGH = 12
POINT_LKNEE = 13


def calc_cos_thigh(seq_id, frame_id):

    with open(f'{INPUT_POSE_ESTIMATION_DIR}{seq_id}/{frame_id}_keypoints.json') as json_file:
        data = json.load(json_file)

    assert len(data['people']) == 1

    x_lthigh = data['people'][0]['pose_keypoints_2d'][POINT_LTHIGH*3]
    y_lthigh = data['people'][0]['pose_keypoints_2d'][POINT_LTHIGH*3 + 1]
    c_lthigh = data['people'][0]['pose_keypoints_2d'][POINT_LTHIGH*3 + 2]
    # print(x_lthigh, y_lthigh, c_lthigh)

    x_lknee = data['people'][0]['pose_keypoints_2d'][POINT_LKNEE*3]
    y_lknee = data['people'][0]['pose_keypoints_2d'][POINT_LKNEE*3 + 1]
    c_lknee = data['people'][0]['pose_keypoints_2d'][POINT_LKNEE*3 + 2]
    # print(x_lknee, y_lknee, c_lknee)

    # angle in rad (- because x-axis is inverted)
    m = - math.atan2(y_lthigh-y_lknee, x_lthigh-x_lknee)

    # angle in degrees
    theta = math.degrees(m)

    # cosinus of angle (between thigh and x-axis)
    cos_theta = math.cos(m)

    return cos_theta


for seq_id in [x for x in os.listdir(INPUT_SEQUENCES_DIR) if os.path.isdir(INPUT_POSE_ESTIMATION_DIR + x)]:

    # retrieve framenames and sort
    frames = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(
        f'{INPUT_SEQUENCES_DIR}{seq_id}/*.png')]
    frames.sort()

    # retrieve keypoints and calculate cos(th) of thigh
    cos_thighs = [calc_cos_thigh(seq_id, f) for f in frames]

    # plot cos(th) per frame
    sns.scatterplot(x=range(len(cos_thighs)),
                    y=cos_thighs,
                    label='subject ' + seq_id)

plt.title('Inclination of Left Thigh')
plt.xlabel('x')
plt.ylabel('cos(θ)')
plt.legend(loc='upper right')
plt.savefig(OUTPUT_DIR + 'plots/inclination_of_left_thigh')
plt.show()
plt.clf()

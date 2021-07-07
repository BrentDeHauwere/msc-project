# %%
import glob
import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.core.fromnumeric import sort
from numpy.linalg import inv
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

INPUT_SEQUENCES_DIR = 'data/processed/sequences/'
INPUT_SILHOUETTES_DIR = 'data/raw/silhouettes/'
INPUT_SEGMENTATION_DIR = 'data/processed/segmentation/'
INPUT_POSE_ESTIMATION_DIR = 'data/processed/pose_estimation/'
INPUT_FLOW_DIR = 'data/processed/flow/'
OUTPUT_DIR = 'data/results/'
FLAG_DEEPFLOW2 = True

FLAG_SILHOUETTES = None
FLAG_SEGMENTATION = None

ACC_MAGNITUDE_THRESHOLD = 1
SCALE_MAGNITUDE = 1

LTHIGH_SEGMENT_COLOUR = [0, 255, 0]  # BGR
LLEG_SEGMENT_COLOUR = [127, 255, 127]  # BGR
TORSO_SEGMENT_COLOUR = [0, 0, 255]  # BGR


def circle_centre(x1, y1, x2, y2, x3, y3):
    """Calculates the coordinates of O, the centre of the circular motion

    :param x1, y1, x2, y2, x3, y3: coordinates of a point in three consecutive frames are: P_i(x_i, y_i) i ∈ (1, 2, 3)
    :type x1, y1, x2, y2, x3, y3: int
    """

    phi = np.array([[x2 - x1, y2 - y1],
                    [x3 - x2, y3 - y2]])
    psi = np.array([[(x2**2 - x1**2 + y2**2 - y1**2)],
                    [(x3**2 - x2**2 + y3**2 - y2**2)]])

    o = 0.5 * inv(phi) @ (psi)  # [2 x 2] @ [2 x 1] =[2 x 1]

    return o[0, 0], o[1, 0]


def rotate(x, y, theta):
    # 3D rotation matrix, source: https://en.wikipedia.org/wiki/Rotation_matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])

    # syntax [x, y, 1]: adding an extra dimension to each vector to enable single transform matrix
    rotated = np.array([[x, y, 1]]) @ rot_matrix

    return rotated[0, 0], rotated[0, 1]


def imageplane_to_cartesian(x, y, mask_rows, mask_cols):
    transf_matrix = np.array([[1, 0, 0],  # aspect ratio
                              [0, -1, 0],  # aspect ratio
                              [-0.5*mask_cols, 0.5*mask_rows, 1]])  # keystone distortions (homography / projective transformation)
    transformed = np.array([[x, y, 1]]) @ transf_matrix

    # TODO: need to divide by transformed[0, 2]? see computer vision slides L8-consistency
    return transformed[0, 0], transformed[0, 1]


def cartesian_to_imageplane(x, y, mask_rows, mask_cols):
    transf_matrix = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0.5*mask_cols, 0.5*mask_rows, 1]])
    transformed = np.array([[x, y, 1]]) @ transf_matrix

    return transformed[0, 0], transformed[0, 1]


def calc_tan_rad(mask_rows, mask_cols, i, j, u1, v1, u2, v2):
    """Calculates the tangential and radial acceleration TODO

    :param mask_rows: number of rows in the mask
    :type mask_rows: int
    :param mask_cols: number of columns in the mask
    :type mask_cols: int
    :param i: i coordinate of the mask
    :type i: int
    :param j: j coordinate of the mask
    :type j: int
    :param u1: horizontal velocity field between frame t and t-1 TODO correct? or t-1 and t?
    :type u1: [type] TODO
    :param v1: vertical velocity field between frame t and t-1
    :type v1: [type] TODO
    :param u2: horizontal velocity field between frame t and t+1
    :type u2: [type] TODO
    :param v2: vertical velocity field between frame t and t+1
    :type v2: [type] TODO
    """

    x, y = imageplane_to_cartesian(j, i, mask_rows, mask_cols)

    back_x = x + u1
    back_y = y - v1
    for_x = x + u2
    for_y = y - v2

    A = np.array([[x-back_x, y-back_y], [for_x-x, for_y-y]])

    # if matrix has full rank (because otherwise it is not invertable)
    if A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]:
        # compute centre of the circular motion
        O_x, O_y = circle_centre(back_x, back_y, x, y, for_x, for_y)

        # set 'OP_2' vector
        PO = np.array([O_x-x, O_y-y])
        PO_magnitude = np.sqrt(PO.dot(PO))

        if PO_magnitude < 0:
            print('ERROR: negative PO magnitute', PO_magnitude)

        sinPOX = (y - O_y) / PO_magnitude
        cosPOX = (x - O_x) / PO_magnitude

        # POX is the angle between OP_2 and the horizontal axis, in degrees (arcsin)
        POX = np.arcsin(sinPOX)
        if cosPOX < 0:
            POX = np.pi - POX

        # Ox_rotated, Oy_rotated = rotate(O_x, O_y, POX) # TODO: unused statement? mistake?

        # 'p_2' represents the coordinates of middle frame
        x_rotated, y_rotated = rotate(x, y, POX)
        # 'a' represents coordinates of acceleration vector
        acc_x, acc_y = rotate(x+u1+u2, y-v1-v2, POX)

        # apply formulas (3.34)
        radial_x, radial_y = rotate(acc_x, y_rotated, -POX)
        tangential_x, tangential_y = rotate(x_rotated, acc_y, -POX)

        # project coordinates back onto the image plane
        radial_j, radial_i = cartesian_to_imageplane(
            radial_x, radial_y, mask_rows, mask_cols)
        tangential_j, tangential_i = cartesian_to_imageplane(
            tangential_x, tangential_y, mask_rows, mask_cols)
        O_j, O_i = cartesian_to_imageplane(O_x, O_y, mask_rows, mask_cols)
    else:
        radial_i = 0
        radial_j = 0
        tangential_i = 0
        tangential_j = 0
        O_i = i
        O_j = j

    return radial_i, radial_j, tangential_i, tangential_j, O_i, O_j


def draw_arrows(img, flow, step=10):
    """Draws a quiver plot. The flow field is visualized by an arrows which start at the initial position points and ends at the new position points, given the displacement.
    Hence, the length of the arrow indicates the magnitude of the displacement.

    :param img: source frame
    :type img: ndarray
    :param flow: optical flow field
    :type flow: ndarray
    :param step: spacing between the arrows, defaults to 16
    :type step: int, optional
    :return: source frame with a quiver plot drawn over it
    :rtype: ndarray
    """

    # retrieve height and length of the source video
    height, width = img.shape[:2]

    # create grid to draw dots and arrays according to the step value; y = [   8    8    8 ...  712  712  712]; x = [   8   24   40 ... 1240 1256 1272]
    y, x = np.mgrid[step/2:height:step, step /
                    2:width:step].reshape(2, -1).astype(int)

    # retrieve the delta x and delta y displacement of the positions on the grid
    dx, dy = flow[y, x].T

    # for each point in the quiver plot, get an array with the x and y position, and the x and y position plus intensity change (flow)
    # these four numbers indicate the start and end of an arrow
    lines = np.vstack([x, y, x+dx, y+dy]).T.reshape(-1, 2, 2)

    # round floating point numbers
    lines = np.int32(lines + 0.5)

    # converts RBG to BGR (opencv's default)
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # colour to use for drawing
    ARROW_COLOUR = (0, 0, 255)

    # draw lines on image vis
    cv2.polylines(vis, lines, 0, ARROW_COLOUR, thickness=1)

    # draw a circle on image vis for every point in the quiver plot
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, ARROW_COLOUR, -1)

    return vis


def draw_hsv(frame, flow):
    """Draws the optical flow field using the HSV colour space

    :param flow: optical flow field
    :type flow: ndarray
    :return: mask of the flow field
    :rtype: ndarray
    """

    # retrieve height and length of the source video
    height, width = flow.shape[:2]

    # delta x and delta y: the displacement from the respective previous frames, both on the x and y axis
    dx, dy = flow[:, :, 0], flow[:, :, 1]

    # computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(dx, dy)

    # creates a mask filled with zero intensities, with the same dimensions as the source frame
    hsv = np.zeros((height, width, 3), np.uint8)  # np.zeroes_like(flow)

    # sets image hue according to the optical flow direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # sets image saturation to maximum
    hsv[..., 1] = 255

    # sets mask value according to the optical flow magnitude (normalized)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # converts HSV (hue saturation value) to BGR (RGB) colour space/representation
    mask_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # converts RGB TO BGR (opencv default)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # linearly blend the hsv mask and the frame
    outp = cv2.addWeighted(mask_bgr, 0.8, frame_bgr, 0.2, 0)

    return outp


def _show_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    plt.show()


if __name__ == '__main__':
    # iterate over each sequence (represented by a directory)
    for seq_id in [x for x in os.listdir(INPUT_SEQUENCES_DIR) if os.path.isdir(INPUT_SEQUENCES_DIR + x)]:

        # create output directories if don't exist
        dirs = ['',
                'velocity',
                'velocity_hsv',
                'acceleration',
                'acceleration_hsv',
                'radial',
                'radial_hsv',
                'tangential',
                'tangential_hsv']
        [os.mkdir(f'{OUTPUT_DIR}{seq_id}/{x}')
         for x in dirs if not os.path.exists(f'{OUTPUT_DIR}{seq_id}/{x}')]

        # retrieve frame filenames and sort
        frames_fname = [os.path.basename(x) for x in glob.glob(
            f'{INPUT_SEQUENCES_DIR}{seq_id}/*.png')]
        frames_fname.sort()

        # extract basename from frame filenames
        frames_basename = [os.path.splitext(f)[0] for f in frames_fname]

        # load frames
        frames = [np.array(Image.open(
            f'{INPUT_SEQUENCES_DIR}{seq_id}/{fn}')) for fn in frames_fname]

        # obtain frame dimensions
        frame_rows, frame_cols, frame_dims = frames[0].shape

        # if provided (look for first), load masks (silhouettes)
        if (FLAG_SILHOUETTES := os.path.isfile(f'{INPUT_SILHOUETTES_DIR}{seq_id}/A{frames_fname[0]}')) \
                and (FLAG_SEGMENTATION := os.path.isfile(f'{INPUT_SEGMENTATION_DIR}{seq_id}/{frames_basename[0]}.npy')):

            # load corresponding silhouettes (note: 0-1 scale // opencv does not support float64, default numpy)
            masks = [np.array(Image.open(f'{INPUT_SILHOUETTES_DIR}{seq_id}/A{fn}'), dtype=np.float32)
                     for fn in frames_fname]

            # obtain mask dimensions
            mask_rows, mask_cols = masks[0].shape

            # obtain coordinates of mask (where intensity is 1 / white)
            mask_coords = [np.argwhere(m == 1) for m in masks]

            # calculate dimensions of each mask
            mask_row_min = [min(mc[:, 0]) for mc in mask_coords]
            mask_row_max = [max(mc[:, 0]) for mc in mask_coords]
            mask_col_min = [min(mc[:, 1]) for mc in mask_coords]
            mask_col_max = [max(mc[:, 1]) for mc in mask_coords]

            # load corresponding segmentation files
            segm_arr = [np.load(f'{INPUT_SEGMENTATION_DIR}{seq_id}/{fbn}.npy')
                        for fbn in frames_basename]

            # only retain the segmentation of the thigh (returns 0 or 255)
            segm_mask_thigh = [cv2.inRange(sa, tuple(LTHIGH_SEGMENT_COLOUR),
                                           tuple(LTHIGH_SEGMENT_COLOUR)) for sa in segm_arr]
            segm_mask_leg = [cv2.inRange(sa, tuple(LLEG_SEGMENT_COLOUR),
                                         tuple(LLEG_SEGMENT_COLOUR)) for sa in segm_arr]
            segm_mask_torso = [cv2.inRange(sa, tuple(TORSO_SEGMENT_COLOUR),
                                           tuple(TORSO_SEGMENT_COLOUR)) for sa in segm_arr]

        # array to save average acceleration in thigh and leg per frame
        rad_thigh_arr = []
        rad_leg_arr = []
        rad_torso_arr = []

        # iterative over all frames
        for frame_i in tqdm(range(len(frames_fname) - 2), desc=seq_id):

            # retrieve next three frames (NOTE: 0-255 scale)
            frame1, frame2, frame3 = frames[frame_i:frame_i+3].copy()

            # if silhouettes are provided (look for first), create a mask
            if FLAG_SILHOUETTES and FLAG_SEGMENTATION:

                # silhouettes
                mask1, mask2, mask3 = masks[frame_i:frame_i+3].copy()

                # dimensions of each mask
                mask1_row_min, mask2_row_min, mask3_row_min = mask_row_min[frame_i:frame_i+3]
                mask1_row_max, mask2_row_max, mask3_row_max = mask_row_max[frame_i:frame_i+3]
                mask1_col_min, mask2_col_min, mask3_col_min = mask_col_min[frame_i:frame_i+3]
                mask1_col_max, mask2_col_max, mask3_col_max = mask_col_max[frame_i:frame_i+3]

                # calculate the union dimensions of the three masks (clip at 0 and frame_rows/cols)
                mask_L_boundary = max(
                    min(mask1_col_min, mask2_col_min, mask3_col_min) - 50,
                    0
                )
                mask_R_boundary = min(
                    max(mask1_col_max, mask2_col_max, mask3_col_max) + 50,
                    frame_cols
                )
                mask_T_boundary = max(
                    min(mask1_row_min, mask2_row_min, mask3_row_min) - 50,
                    0
                )
                mask_B_boundary = min(
                    max(mask1_row_max, mask2_row_max, mask3_row_max) + 50,
                    frame_rows
                )

                # segmentation of the body parts (returns 0 or 255)
                segm_mask1_thigh, segm_mask2_thigh, segm_mask3_thigh = segm_mask_thigh[
                    frame_i:frame_i+3]
                segm_mask1_leg, segm_mask2_leg, segm_mask3_leg = segm_mask_leg[frame_i:frame_i+3]
                segm_mask1_torso, segm_mask2_torso, segm_mask3_torso = segm_mask_torso[
                    frame_i:frame_i+3]

                # remove segmentation errors by filtering out what is out of the silhouette mask
                mask_select = np.ones((mask_rows, mask_cols), bool)
                mask_select[mask_T_boundary:mask_B_boundary,
                            mask_L_boundary: mask_R_boundary] = 0
                segm_mask2_thigh[mask_select] = 0
                segm_mask2_leg[mask_select] = 0
                segm_mask2_torso[mask_select] = 0

                # _show_image(cv2.addWeighted(
                #     cv2.cvtColor(segm_mask1, cv2.COLOR_GRAY2RGB), 0.6,
                #     cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR), 0.4, 0.0), 'mask_segm')

            else:

                # if no no mask specified, use frame dimensions
                mask_L_boundary = 0
                mask_R_boundary = frame_cols
                mask_T_boundary = 0
                mask_B_boundary = frame_rows

                # mask dimensions = frame dimensions
                mask_rows, mask_cols, _ = frame1.shape

            # indicates whether flow is already computed with deepflow2
            if FLAG_DEEPFLOW2 == False:

                # search areas in frames
                cropped_frame1 = frame1[mask_T_boundary:mask_B_boundary,
                                        mask_L_boundary: mask_R_boundary]
                cropped_frame2 = frame2[mask_T_boundary: mask_B_boundary,
                                        mask_L_boundary: mask_R_boundary]
                cropped_frame3 = frame3[mask_T_boundary: mask_B_boundary,
                                        mask_L_boundary: mask_R_boundary]

                # creates object to compute optical flow by DeepFlow method, version 1
                opt_flow = cv2.optflow.createOptFlow_DeepFlow()

                # velocity field V(t ~ t-∆t)
                opt1 = opt_flow.calc(
                    cv2.cvtColor(cropped_frame2, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(cropped_frame1, cv2.COLOR_BGR2GRAY),
                    None
                )

                # velocity field V(t ~ t+∆t)
                opt2 = opt_flow.calc(
                    cv2.cvtColor(cropped_frame2, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(cropped_frame3, cv2.COLOR_BGR2GRAY),
                    None
                )

                # horizontal (dx) and vertical (dy) displacement (~velocity field)
                # pad the velocity fields (outside search area) with 0's to have the same size as the frame again
                u1 = np.pad(opt1[:, :, 0], ((mask_T_boundary, frame_rows-mask_B_boundary),
                                            (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
                v1 = np.pad(opt1[:, :, 1], ((mask_T_boundary, frame_rows-mask_B_boundary),
                                            (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
                u2 = np.pad(opt2[:, :, 0], ((mask_T_boundary, frame_rows-mask_B_boundary),
                                            (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
                v2 = np.pad(opt2[:, :, 1], ((mask_T_boundary, frame_rows-mask_B_boundary),
                                            (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)

            else:

                # velocity field V(t ~ t-∆t) and V(t ~ t+∆t), computed by DeepFlow2
                opt1 = np.load(
                    f'{INPUT_FLOW_DIR}{seq_id}/{frame_i+1:06d}_{frame_i:06d}.npy')
                opt2 = np.load(
                    f'{INPUT_FLOW_DIR}{seq_id}/{frame_i+1:06d}_{frame_i+2:06d}.npy')

                # horizontal (dx) and vertical (dy) displacement (~velocity field)
                u1 = opt1[:, :, 0]
                v1 = opt1[:, :, 1]
                u2 = opt2[:, :, 0]
                v2 = opt2[:, :, 1]

            # compute horizontal and vertical acceleration
            acc_u = u2 + u1
            acc_v = v2 + v1

            #  array to store tangential and radial acceleration in each pixel
            radial = np.zeros((frame_rows, frame_cols, 2))
            tangential = np.zeros((frame_rows, frame_cols, 2))

            # for each pixel in the search area, calculate the radial acceleration
            # TODO: optimise with vectorisation
            for i in range(mask_T_boundary, mask_B_boundary):
                for j in range(mask_L_boundary, mask_R_boundary):

                    # compute magnitude of acceleration vector (for one pixel)
                    acc_mag = np.sqrt(acc_u[i, j]**2 + acc_v[i, j]**2)

                    # if magnitude of acceleration vector is large enough
                    if acc_mag > ACC_MAGNITUDE_THRESHOLD:
                        # calculate radial and tangential acceleration, and circle centre
                        radial_i, radial_j, tangential_i, tangential_j, O_i, O_j = calc_tan_rad(mask_rows, mask_cols, i, j,
                                                                                                u1[i, j], v1[i, j], u2[i, j], v2[i, j])

                        # save radial and tangential acceleration in designated array
                        radial[i, j, 0] = radial_j - j
                        radial[i, j, 1] = radial_i - i
                        tangential[i, j, 0] = tangential_j - j
                        tangential[i, j, 1] = tangential_i - i

            # if segmentations are provided (look for first), analyse acceleration in body parts
            if FLAG_SEGMENTATION:
                rad_thigh_arr.append(np.mean(
                    np.sqrt(
                        radial[segm_mask2_thigh == 255, 0]**2
                        + radial[segm_mask2_thigh == 255, 1]**2
                    )
                ))

                rad_leg_arr.append(np.mean(
                    np.sqrt(
                        radial[segm_mask2_leg == 255, 0]**2
                        + radial[segm_mask2_leg == 255, 1]**2
                    )
                ))

                rad_torso_arr.append(np.mean(
                    np.sqrt(
                        radial[segm_mask2_torso == 255, 0]**2
                        + radial[segm_mask2_torso == 255, 1]**2
                    )
                ))

            # # draw plots (and scale magnitude to make arrows more visible)
            # vel_viz_hsv = draw_hsv(frame2, flow=np.stack(
            #     [u2, v2], axis=-1)*SCALE_MAGNITUDE)
            # acc_viz_hsv = draw_hsv(frame2, flow=np.stack(
            #     [acc_u, acc_v], axis=-1)*SCALE_MAGNITUDE)
            # rad_viz_hsv = draw_hsv(frame2, flow=radial*SCALE_MAGNITUDE)
            # tan_viz_hsv = draw_hsv(frame2, flow=tangential*SCALE_MAGNITUDE)

            # vel_viz = draw_arrows(frame2, flow=np.stack(
            #     [u2, v2], axis=-1)*SCALE_MAGNITUDE)
            # acc_viz = draw_arrows(
            #     frame2, flow=np.stack([acc_u, acc_v], axis=-1)*SCALE_MAGNITUDE)
            # rad_viz = draw_arrows(frame2, flow=radial*SCALE_MAGNITUDE)
            # tan_viz = draw_arrows(frame2, flow=tangential*SCALE_MAGNITUDE)

            # # save plots
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/velocity_hsv/{frame_i+1:06d}.png', vel_viz_hsv)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/acceleration_hsv/{frame_i+1:06d}.png', acc_viz_hsv)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/radial_hsv/{frame_i+1:06d}.png', rad_viz_hsv)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/tangential_hsv/{frame_i+1:06d}.png', tan_viz_hsv)

            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/velocity/{frame_i+1:06d}.png', vel_viz)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/acceleration/{frame_i+1:06d}.png', acc_viz)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/radial/{frame_i+1:06d}.png', rad_viz)
            # cv2.imwrite(
            #     f'{OUTPUT_DIR}{seq_id}/tangential/{frame_i+1:06d}.png', tan_viz)

        # save average radial acceleration in body parts
        np.save(f'{OUTPUT_DIR}{seq_id}_rad_thigh.npy',
                np.array(rad_thigh_arr))
        np.save(f'{OUTPUT_DIR}{seq_id}_rad_leg.npy',
                np.array(rad_leg_arr))
        np.save(f'{OUTPUT_DIR}{seq_id}_rad_torso.npy',
                np.array(rad_torso_arr))

import os
import cv2

import numpy as np
from numpy.core.fromnumeric import sort
from numpy.linalg import inv
from PIL import Image
from tqdm import tqdm

INPUT_SEQUENCES_DIR = 'data/processed/sequences/'
INPUT_SILHOUETTES_DIR = 'data/raw/silhouettes/'
OUTPUT_DIR = 'data/results/'

ACC_MAGNITUDE_THRESHOLD = 1


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

# TODO: ask for reason?


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


if __name__ == '__main__':
    # iterate over each sequence
    for dirpath, dirnames, filenames in os.walk(INPUT_SEQUENCES_DIR):
        print('Processing:', dirpath)

        # id of the sequence video
        seq_id = os.path.basename(dirpath)

        # sort sequence frames
        frames = sort(filenames)

        # iterative over all frames
        for frame_i in tqdm(range(len(frames) - 2), desc=seq_id):

            # retrieve next three frames (NOTE: 0-255 scale)
            frame1 = np.array(Image.open(
                f'{INPUT_SEQUENCES_DIR}{seq_id}/{frames[frame_i]}'))
            frame2 = np.array(Image.open(
                f'{INPUT_SEQUENCES_DIR}{seq_id}/{frames[frame_i+1]}'))
            frame3 = np.array(Image.open(
                f'{INPUT_SEQUENCES_DIR}{seq_id}/{frames[frame_i+2]}'))

            # obtain frame dimensions
            frame_rows, frame_cols, frame_dims = frame1.shape

            # retrieve corresponding silhouettes (NOTE: 0-1 scale)
            mask1 = np.array(Image.open(
                f'{INPUT_SILHOUETTES_DIR}{seq_id}/A{frames[frame_i]}'), dtype=float)
            mask2 = np.array(Image.open(
                f'{INPUT_SILHOUETTES_DIR}{seq_id}/A{frames[frame_i+1]}'), dtype=float)
            mask3 = np.array(Image.open(
                f'{INPUT_SILHOUETTES_DIR}{seq_id}/A{frames[frame_i+2]}'), dtype=float)

            # obtain mask dimensions
            mask_rows, mask_cols = mask1.shape

            # obtain coordinates of mask (where intensity is 1)
            mask_coords1 = np.argwhere(mask1 == 1)
            mask_coords2 = np.argwhere(mask2 == 1)
            mask_coords3 = np.argwhere(mask3 == 1)

            # calculate dimensions of each mask
            mask1_width_min = min(mask_coords1[:, 0])
            mask1_width_max = max(mask_coords1[:, 0])
            mask1_height_min = min(mask_coords1[:, 1])
            mask1_height_max = max(mask_coords1[:, 1])

            mask2_width_min = min(mask_coords2[:, 0])
            mask2_width_max = max(mask_coords2[:, 0])
            mask2_height_min = min(mask_coords2[:, 1])
            mask2_height_max = max(mask_coords2[:, 1])

            mask3_width_min = min(mask_coords3[:, 0])
            mask3_width_max = max(mask_coords3[:, 0])
            mask3_height_min = min(mask_coords3[:, 1])
            mask3_height_max = max(mask_coords3[:, 1])

            # calculate the union dimensions of the three masks (clip at 0 and frame_rows/cols)
            mask_L_boundary = max(
                min(mask1_width_min, mask2_width_min, mask3_width_min) - 50,
                0
            )
            mask_R_boundary = min(
                max(mask1_width_max, mask2_width_max, mask3_width_max) + 50,
                frame_cols
            )
            mask_T_boundary = max(
                min(mask1_height_min, mask2_height_min, mask3_height_min) - 50,
                0
            )
            mask_B_boundary = min(
                max(mask1_height_max, mask2_height_max, mask3_height_max) + 50,
                frame_rows
            )

            # region of interest in frames
            cropped_frame1 = frame1[mask_T_boundary:mask_B_boundary,
                                    mask_L_boundary:mask_R_boundary]
            cropped_frame2 = frame2[mask_T_boundary:mask_B_boundary,
                                    mask_L_boundary:mask_R_boundary]
            cropped_frame3 = frame3[mask_T_boundary:mask_B_boundary,
                                    mask_L_boundary:mask_R_boundary]

            # creates object to compute optical flow by DeepFlow method
            # TODO: replace with DeepFlow2 and use RBG cropped_frameX
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
            u1 = opt1[:, :, 0]
            v1 = opt1[:, :, 1]
            u2 = opt2[:, :, 0]
            v2 = opt2[:, :, 1]

            # for each pixel in the cropped mask
            # TODO: optimise with vectorisation
            for i in range(mask_T_boundary, mask_B_boundary):
                for j in range(mask_L_boundary, mask_R_boundary):

                    # compute horizontal and vertical acceleration (for one pixel)
                    acc_u = u2[i, j] + u1[i, j]
                    acc_v = v2[i, j] + v1[i, j]

                    # compute magnitude of acceleration vector
                    acc_mag = np.sqrt(acc_u**2 + acc_v**2)

                    if acc_mag > ACC_MAGNITUDE_THRESHOLD:
                        radial_i, radial_j, tangential_i, tangential_j, O_i, O_j = calc_tan_rad(mask_rows, mask_cols, i, j,
                                                                                                u1[i, j], v1[i, j], u2[i, j], v2[i, j])
                        pass

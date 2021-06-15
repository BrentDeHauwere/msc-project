# %%
import glob
import os

import cv2
import numpy as np
from numpy.core.fromnumeric import sort
from numpy.linalg import inv
from PIL import Image, ImageDraw
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


def draw_hsv(flow):
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
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


if __name__ == '__main__':
    # iterate over each sequence (represented by a directory)
    for seq_id in [x for x in os.listdir(INPUT_SEQUENCES_DIR) if os.path.isdir(INPUT_SEQUENCES_DIR + x)]:

        # create output directories if don't exist
        dirs = ['',
                'velocity_quiver',
                'velocity_hsv',
                'acceleration',
                'radial',
                'tangential']
        [os.mkdir(f'{OUTPUT_DIR}{seq_id}/{x}')
         for x in dirs if not os.path.exists(f'{OUTPUT_DIR}{seq_id}/{x}')]

        # retrieve frame filenames and sort
        frames = [os.path.basename(x) for x in glob.glob(
            f'{INPUT_SEQUENCES_DIR}{seq_id}/*.png')]
        frames.sort()

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
            mask1_row_min = min(mask_coords1[:, 0])
            mask1_row_max = max(mask_coords1[:, 0])
            mask1_col_min = min(mask_coords1[:, 1])
            mask1_col_max = max(mask_coords1[:, 1])

            mask2_row_min = min(mask_coords2[:, 0])
            mask2_row_max = max(mask_coords2[:, 0])
            mask2_col_min = min(mask_coords2[:, 1])
            mask2_col_max = max(mask_coords2[:, 1])

            mask3_row_min = min(mask_coords3[:, 0])
            mask3_row_max = max(mask_coords3[:, 0])
            mask3_col_min = min(mask_coords3[:, 1])
            mask3_col_max = max(mask_coords3[:, 1])

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

            # search areas in frames
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

            # pad the velocity fields (outside search area) with 0's to have the same size as the frame again
            u1 = np.pad(u1, ((mask_T_boundary, frame_rows-mask_B_boundary),
                        (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
            v1 = np.pad(v1, ((mask_T_boundary, frame_rows-mask_B_boundary),
                        (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
            u2 = np.pad(u2, ((mask_T_boundary, frame_rows-mask_B_boundary),
                        (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)
            v2 = np.pad(v2, ((mask_T_boundary, frame_rows-mask_B_boundary),
                        (mask_L_boundary, frame_cols-mask_R_boundary)), 'constant', constant_values=0)

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

            # draw and save quiver plot of velocity field (between second and third frame, on third frame)
            vel_viz_arr = draw_arrows(frame3, flow=np.stack([u2, v2], axis=-1))
            cv2.imwrite(
                f'{OUTPUT_DIR}{seq_id}/velocity_quiver/{frame_i+2:06d}.png', vel_viz_arr)

            # draw and save hsv plot of velocity field
            vel_viz_hsv = draw_hsv(flow=np.stack([u2, v2], axis=-1))
            cv2.imwrite(
                f'{OUTPUT_DIR}{seq_id}/velocity_hsv/{frame_i+2:06d}.png', vel_viz_hsv)

            # draw and save acceleration field
            acc_viz = draw_hsv(flow=np.stack([acc_u, acc_v], axis=-1))
            cv2.imwrite(
                f'{OUTPUT_DIR}{seq_id}/acceleration/{frame_i+2:06d}.png', acc_viz)

            # draw and save radial acceleration
            acc_viz_rad = draw_arrows(frame3, flow=radial)
            cv2.imwrite(
                f'{OUTPUT_DIR}{seq_id}/radial/{frame_i+2:06d}.png', acc_viz_rad)

            # draw and save tangential acceleration
            acc_viz_tan = draw_arrows(frame3, flow=tangential)
            cv2.imwrite(
                f'{OUTPUT_DIR}{seq_id}/tangential/{frame_i+2:06d}.png', acc_viz_tan)

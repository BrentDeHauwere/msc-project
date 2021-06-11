import numpy as np
from numpy.linalg import inv


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


def imageplane_to_cartesian(x, y, mask_m_rows, mask_n_cols):
    transf_matrix = np.array([[1, 0, 0],  # aspect ratio
                              [0, -1, 0],  # aspect ratio
                              [-0.5*mask_n_cols, 0.5*mask_m_rows, 1]])  # keystone distortions (homography / projective transformation)
    transformed = np.array([[x, y, 1]]) @ transf_matrix

    # TODO: need to divide by transformed[0, 2]? see computer vision slides L8-consistency
    return transformed[0, 0], transformed[0, 1]


def cartesian_to_imageplane(x, y, mask_m_rows, mask_n_cols):
    transf_matrix = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0.5*mask_n_cols, 0.5*mask_m_rows, 1]])
    transformed = np.array([[x, y, 1]]) @ transf_matrix

    return transformed[0, 0], transformed[0, 1]


def calc_tan_rad(mask_m_rows, mask_n_cols, i, j, u1, v1, u2, v2):
    """Calculates the tangential and radial acceleration TODO

    :param mask_m_rows: number of rows in the mask
    :type mask_m_rows: int
    :param mask_n_cols: number of columns in the mask
    :type mask_n_cols: int
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

    x, y = imageplane_to_cartesian(j, i, mask_m_rows, mask_n_cols)

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
            radial_x, radial_y, mask_m_rows, mask_n_cols)
        tangential_j, tangential_i = cartesian_to_imageplane(
            tangential_x, tangential_y, mask_m_rows, mask_n_cols)
        O_j, O_i = cartesian_to_imageplane(O_x, O_y, mask_m_rows, mask_n_cols)
    else:
        radial_i = 0
        radial_j = 0
        tangential_i = 0
        tangential_j = 0
        O_i = i
        O_j = j

    return radial_i, radial_j, tangential_i, tangential_j, O_i, O_j

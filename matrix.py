import numpy as np
#Транспонированная матрица
def T(a,b,c):
    A = np.array([
         [1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [a,b,c,1]
                 ])
    return A

def vector_length(end , begin):
    length = end - begin
    length = np.sqrt(length*length)
    return length
###
def rot_matrix(theta , c):
    result = np.zeros((4, 4))
    result[0, 0] = np.cos(theta) + (c[0] ** 2) * (1 - np.cos(theta))
    result[0, 1] = c[0] * c[1] * (1 - np.cos(theta)) - c[2] * np.sin(theta)
    result[0, 2] = c[0] * c[2] * (1 - np.cos(theta)) + c[1] * np.sin(theta)
    result[1, 0] = c[0] * c[1] * (1 - np.cos(theta)) + c[2] * np.sin(theta)
    result[1, 1] = np.cos(theta) + (c[1] ** 2) * (1 - np.cos(theta))
    result[1, 2] = c[1] * c[2] * (1 - np.cos(theta)) - c[0] * np.sin(theta)
    result[2, 0] = c[0] * c[2] * (1 - np.cos(theta)) - c[1] * np.sin(theta)
    result[2, 1] = c[1] * c[2] * (1 - np.cos(theta)) + c[0] * np.sin(theta)
    result[2, 2] = np.cos(theta) + (c[2] ** 2) * (1 - np.cos(theta))
    result[3, 3] = 1
    return result

#
def rotation(point, vector_begin, vector_end, theta):

    c = (vector_end - vector_begin) / vector_length(vector_end , vector_begin)
    rot = point_to_rot(point)
    rot_shift = rot @ T(-1 * vector_begin)
    rot_shift_rot = rot_shift @ rot_matrix(theta, c)
    rot_rot = rot_shift_rot @ T(vector_begin)
    point_l = rot_to_point(rot_rot)
    return point_l

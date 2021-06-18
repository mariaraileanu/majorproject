from pylab import *
from scipy.interpolate import interp1d
from skimage import color
import numpy as np
from numpy import *
from cv2 import *
import matplotlib.pyplot as plt
import cv2




x = []  # will contain the x coordinates of points on lips
y = []  # will contain the y coordinates of points on lips


def inter(lx, ly, k1='quadratic'):
    unew = np.arange(lx[0], lx[-1] + 1, 1)
    f2 = interp1d(lx, ly, kind=k1, fill_value="extrapolate")
    return f2, unew


def ext(a, b, i):

    global x
    global y

    if a > b:
        temp = a
        a = b
        b = temp

    a, b = np.round(a), np.round(b)
    x.extend(arange(a, b, 1, dtype=np.int32).tolist())
    y.extend((ones(int(b - a), dtype=np.int32) * i).tolist())

    

def apply_lipstick(frame, points, color_val):

    global x
    global y

    x = []
    y = []
    b, g, r = color_val  # lipstick color

    up_left_end = 3
    up_right_end = 5




    # gets the points on the boundary of lips from the file
    points = np.floor(points)
    point_out_x = np.array((points[:len(points) // 2][:, 1]))
    point_out_y = np.array(points[:len(points) // 2][:, 0])
    point_in_x = (points[len(points) // 2:][:, 1])
    point_in_y = points[len(points) // 2:][:, 0]

    #figure()
    im = frame

    # Code for the curves bounding the lips
    o_u_l = inter(point_out_x[:up_left_end], point_out_y[:up_left_end])
    o_u_r = inter(point_out_x[up_left_end - 1:up_right_end], point_out_y[up_left_end - 1:up_right_end])
    o_l = inter([point_out_x[0]] + point_out_x[up_right_end - 1:][::-1].tolist(),
                [point_out_y[0]] + point_out_y[up_right_end - 1:][::-1].tolist(), 'cubic')
    i_u_l = inter(point_in_x[:up_left_end], point_in_y[:up_left_end])
    i_u_r = inter(point_in_x[up_left_end - 1:up_right_end], point_in_y[up_left_end - 1:up_right_end])
    i_l = inter([point_in_x[0]] + point_in_x[up_right_end - 1:][::-1].tolist(),
                [point_in_y[0]] + point_in_y[up_right_end - 1:][::-1].tolist(), 'cubic')

    for i in range(int(o_u_l[1][0]), int(i_u_l[1][0] + 1)):
        ext(o_u_l[0](i), o_l[0](i) + 1, i)

    for i in range(int(i_u_l[1][0]), int(o_u_l[1][-1] + 1)):
        ext(o_u_l[0](i), i_u_l[0](i) + 1, i)
        ext(i_l[0](i), o_l[0](i) + 1, i)

    for i in range(int(i_u_r[1][-1]), int(o_u_r[1][-1] + 1)):
        ext(o_u_r[0](i), o_l[0](i) + 1, i)

    for i in range(int(i_u_r[1][0]), int(i_u_r[1][-1] + 1)):
        ext(o_u_r[0](i), i_u_r[0](i) + 1, i)
        ext(i_l[0](i), o_l[0](i) + 1, i)

    # Now x and y contains coordinates of all the points on lips

    val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    val[:, 0] += ll
    val[:, 1] += aa
    val[:, 2] += bb

    im[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
    plt.gca().set_aspect('equal', adjustable='box')
    #imshow(im)
    #show()
    #cv2.imwrite('output1.jpg', im)
    return frame



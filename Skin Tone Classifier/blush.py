from scipy import interpolate
from pylab import *
from skimage import color
import numpy as np
from numpy import *
from cv2 import *
import matplotlib.pyplot as plt
import cv2

Bg, Gg, Rg = (223., 91., 111.)
im = []
points = []
height = 0
width = 0
imOrg = []
intensity = 0.0
# mid is used to construct the points for the blush in
# the left cheek(by mirroring the points), because only the right cheek's points are
# given as input.


def blushing(frames, points, mid, i_val):
    
    global height
    global width
    global im
    global imOrg
    global intensity

    intensity = i_val

    im = frames
    points = points

    height, width = im.shape[:2]
    imOrg = im.copy()


    x, y = points[0:5, 0], points[0:5, 1]
    x, y = get_boundary_points(x, y)
    x, y = get_interior_points(x, y)
    apply_blush_color()
    smoothen_blush(x, y)
    smoothen_blush(2 * mid * ones(len(x)) - x, y)

    #plt.figure()
    #plt.imshow(imOrg)

    return imOrg



# In blushing1, I tried to draw separate blushes for each cheek in order to fix blush for rotated faces but it did not work that good
# def blushing1(frames, right_points, left_points, mid, i_val):
    
#     global height
#     global width
#     global im
#     global imOrg
#     global intensity

#     intensity = i_val

#     im = frames
#     right_points = right_points
#     left_points = left_points

#     height, width = im.shape[:2]
#     imOrg = im.copy()


#     x_r, y_r = right_points[0:5, 0], right_points[0:5, 1]
#     x_l, y_l = left_points[0:5, 0], left_points[0:5, 1]
#     x_r, y_r = get_boundary_points(x_r, y_r)
#     x_r, y_r = get_interior_points(x_r, y_r)
#     x_l, y_l = get_boundary_points(x_l, y_l)
#     x_l, y_l = get_interior_points(x_l, y_l)
    
#     apply_blush_color()
#     smoothen_blush(x_r, y_r)
#     smoothen_blush(x_l, y_l)

#     #plt.figure()
#     #plt.imshow(imOrg)

#     return imOrg

def get_boundary_points(x, y):
    tck, u = interpolate.splprep([x, y], s=0, per=1)
    unew = np.linspace(u.min(), u.max(), 1000)
    xnew, ynew = interpolate.splev(unew, tck, der=0)
    tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
    coord = list(set(tuple(map(tuple, tup))))
    coord = np.array([list(elem) for elem in coord])
    return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

def get_interior_points(x, y):
    intx = []
    inty = []

    def ext(a, b, i):
        a, b = round(a), round(b)
        intx.extend(arange(a, b, 1).tolist())
        inty.extend((ones(b - a) * i).tolist())

    x, y = np.array(x), np.array(y)
    xmin, xmax = amin(x), amax(x)
    xrang = np.arange(xmin, xmax + 1, 1)
    for i in xrang:
        ylist = y[where(x == i)]
        ext(amin(ylist), amax(ylist), i)
    return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)


def apply_blush_color(r=Rg, g=Gg, b=Bg):
    global im
    global height
    global width
    
    val = color.rgb2lab((im / 255.)).reshape(width * height, 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll, aa, bb = (L1 - L) * intensity, (A1 - A) * intensity, (B1 - B) * intensity
    val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
    val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
    val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)
    
    im = color.lab2rgb(val.reshape(height, width, 3)) * 255
   

def smoothen_blush(x, y):

    global im
    global imOrg
    global height
    global width

    imgBase = zeros((height, width))
    cv2.fillConvexPoly(imgBase, np.array(c_[x, y], dtype='int32'), 1)
    imgMask = cv2.GaussianBlur(imgBase, (51, 51), 0)
    imgBlur3D = np.ndarray([height, width, 3], dtype='float')
    imgBlur3D[:, :, 0] = imgMask
    imgBlur3D[:, :, 1] = imgMask
    imgBlur3D[:, :, 2] = imgMask
    imOrg = (imgBlur3D * im + (1 - imgBlur3D) * imOrg).astype('uint8')




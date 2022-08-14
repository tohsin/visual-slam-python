# from threading import Thread, Lock

# mutex = Lock()

# def processData(data):
#     mutex.acquire()
#     try:
#         print('Do some stuff')
#     finally:
#         mutex.release()

# while True:
#     t = Thread(target = processData, args = (some_data,))
#     t.start()


'''
Implimentation of dense mapping using NCC, block matching, epipoloar line search

The data set is from monocular camera under known tracjectory
'''

# declare cosnants

from typing import List
import cv2 as cv
import sophus as sp
import numpy as np

BOARDER = 20
WIDTH = 640
HEIGHT = 480
# camera intristics K {fx , fy , cx , cy}
FX: float =  481.2
FY : float  = -480.0
CX : float = 319.5
CY : float = 239.5
NCC_WINDOW_SIZE = 3 # half size for block matching

NCC_AREA = (2 * NCC_WINDOW_SIZE + 1) ** 2 # area of NCC
MIN_CONV = 0.1 # minimal convariance 
MAX_CONV = 10 # max covariance


def px2cam(px) ->np.ndarray:
    '''
    arguments:
        px : numpy arrray representing a point
        flip formula to X,Y,Z form u,v
    returns X , Y , Z
    '''
    return np.array([(px[0] - CX) / FX, (px[1] - CY) / FY, 1])
        
def cam2px(p_cam):
    ''' 
    parameters:
        p_cam : 3d vector u , v
        u = X * FX / Z + cx
        v = Y * FY / Z + CY
    returns u , v
    '''
    return  np.array([(p_cam[0] * FX / p_cam[2]) + CX, (p_cam[1] * FY / p_cam[2]) + CY])


def epipolar_search(ref: cv.Mat, curr : cv.Mat, T_C_R : sp.SE3(),
        pt_ref : List, depth_mu : float, depth_cov : float,
        pt_curr : List, epipolar_direction : List):
        '''
        Parameters
        pt_ref : point in refrence
        '''

        f_ref = px2cam(pt_ref) #3d vector X,Y,Z
        # normalise vector
        f_ref = f_ref / np.sqrt(np.sum(f_ref **2 ))
        P_ref = f_ref * depth_mu # refrecne vector

        pixel_mean_curr = cam2px(T_C_R * P_ref) #pixel according to mean depth

        d_min = depth_mu - 3 * depth_cov
        d_max = depth_mu + 3 * depth_cov
        if d_min < 0.1:
            d_min = 0.1

        pixel_min_curr =  cam2px(T_C_R * ( f_ref * d_min )) # pixel of minimal depth
        pixel_max_curr =  cam2px(T_C_R * ( f_ref * d_max )) # pixel of minimal depth








def update(ref: cv.Mat, curr : cv.Mat, T_C_R : sp.SE3(), depth : cv.Mat,
                     depth_conv2 :cv.Mat):
    '''
    Parameters
        refrence : refrence image
        curr : Current Image
        T_C_R : matrix from refrence to Current
        depth : mean of depth mu
        deptH_conv2 : covariance of depth or std
    '''
    for x in range(BOARDER, WIDTH-BOARDER, 1):
        for y in range(BOARDER, WIDTH-BOARDER, 1):
            if depth_conv2[y][x] < MIN_CONV or depth_conv2 > MAX_CONV:
                # algorithm has converged or abort
                continue
            
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

from os import X_OK
from tkinter import Widget
from turtle import width
from typing import List
from xmlrpc.client import Boolean
import cv2 as cv
import sophus as sp
import numpy as np
import math
import os
from numpy.linalg import norm

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
        u = K * P = fx 0  cx   * X
        v           0  fy cy     y
                    0   0  0     z
        u = X * FX / Z + cx
        v = Y * FY / Z + CY
    returns u , v
    '''
    return  np.array([(p_cam[0] * FX / p_cam[2]) + CX, 
                        (p_cam[1] * FY / p_cam[2]) + CY])

def inside(pt):

    return pt[0] >= BOARDER and pt[1] >= BOARDER and\
        pt[0] + BOARDER < WIDTH and pt[1]+BOARDER <= HEIGHT 

def getBilinearInterpolatedValue(img : cv.Mat , pt ):
    '''
    best explanation first read wikipedia
    you have four pointa surrounding x,y 
    (x,y)             (x+1, y)


             (x_o, y_o)

    (x,y+)             (x+1, y+1)
    and you want to use the four values around
    x_o , y_o to estimate
    a is the distance from x,y to x_o ,y_o in x axis
    and same for b.
    if 

    '''
    x = pt[0]
    y = pt[1]
    x_prime = math.floor(x)
    y_prime = math.floor(y)
    b = y - y_prime#y
    a = x - x_prime
    return  (( (1-a)*(1-b) * img[y_prime][x_prime]) +\
            ((1-a) *  b * (img[y_prime+1][x_prime])) +\
            (a * (1-b)* img[y_prime][x_prime+1]) +\
            (a * b * img[y_prime+1][x_prime+1]))/255.0



def NCC(ref_image : cv.Mat, curr_image : cv.Mat, pt_ref, pt_curr):
    '''
    parameters:
        pt_ref  : refrence point u,v
        pt_curr : current point on image u,v

    '''
    mean_ref = 0
    mean_curr = 0
    values_ref = []
    values_curr = []
    for  x in range(-NCC_WINDOW_SIZE, NCC_WINDOW_SIZE + 1, 1 ):
        for y in range(-NCC_WINDOW_SIZE, NCC_WINDOW_SIZE + 1 ,1):
            value_ref = ref_image[ int(y + pt_ref[1]) ][ int(x + pt_ref[0]) ] /255.0

            mean_ref += value_ref

            value_curr = getBilinearInterpolatedValue(curr_image, np.add(pt_curr, np.array([x,y])))

            mean_curr += value_curr

            values_ref.append(value_ref)
            values_curr.append(value_curr)

    mean_ref /= NCC_AREA
    mean_curr /= NCC_AREA

    # compute Zero mean NCC
    '''
    E stands for sigma
    NCC(A,B) = E_i_j (A_i_j - A_i_j_bar) (B_i_j-B_i_j_bar) / SQRT(E(A_i_j - A_i_J_bar)**2 E(B_i_j - B_i_j_bar)**2)

    '''
    numerator, denominator1, denominator2 = 0, 0, 0
    for i in range(len(values_ref)):
        numerator += (values_ref[i] - mean_ref) * (values_curr[i]- mean_curr) 
    
        denominator1 +=  (values_ref[i]- mean_ref) **2
      
        denominator2 += (values_curr[i] - mean_curr) **2
        
    return numerator / math.sqrt(denominator1 * denominator2 + 1e-10)

def epipolar_search(
        ref_image: cv.Mat, 
        curr_image : cv.Mat,
        T_C_R : sp.SE3(),
        pt_ref : List, 
        depth_mu : float, 
        depth_cov : float,
        pt_curr, 
        epipolar_direction):
        '''
        Parameters
            pt_ref : point in refrence
            T_C_R : Transformation from refence to current
        returns 
            epipoloar_direction
        '''
        # conver point in refrencnce or pixel to 3d Space X, Y, Z
        f_ref = px2cam(pt_ref) #3d vector X,Y,Z 
        # normalise vector
        f_ref = f_ref / np.sqrt(np.sum(f_ref **2 ))
        
        P_ref = f_ref * depth_mu # refrecne vector

        #we then perform R and t on the 3d point and convert the to pixel to get equivalent after motion
        pixel_mean_curr = cam2px(T_C_R * P_ref) #pixel according to mean depth

        #using raduis of mu +- 3sigma as radiius to get the estimate
        # depth_mu was mean value so we get the max and min from that using 3 as max or min std
        d_min = depth_mu - 3 * depth_cov 
        d_max = depth_mu + 3 * depth_cov
        if d_min < 0.1:
            d_min = 0.1
        # convert the min and max points on xyz to refrence image by using transformation T
        pixel_min_curr =  cam2px(T_C_R * ( f_ref * d_min )) # pixel of minimal depth
        pixel_max_curr =  cam2px(T_C_R * ( f_ref * d_max )) # pixel of minimal depth


        # line goes from max to min vector subtraction magnitude
        epipolar_line = pixel_max_curr - pixel_min_curr # epiploar line obtained from max and min
        epipolar_direction = epipolar_line
        epipolar_direction = epipolar_direction  / np.sqrt(np.sum(epipolar_direction **2 ))
        half_length = 0.5 * norm(epipolar_line) #fix

        if half_length>100:
            half_length = 100
        
        #epipolar search for best ncc score
        best_ncc = -1.0
        best_px_curr = None
        l = -half_length
        # while(l<=half_length):
        for l in range(-half_length,half_length +1, 0.7):
         # using a strp of .7 or sqrt 2
            px_curr = pixel_mean_curr + l * epipolar_direction  #point to be matched

            if not inside(px_curr):
                continue
               
            # compute NCC score
            ncc = NCC(ref_image=ref_image, curr_image=curr_image, pt_ref=pt_ref, pt_curr=px_curr)
            if ncc > best_ncc:
                # update if better
                best_ncc = ncc
                best_px_curr = px_curr
        if best_ncc < 0.85:
            return False , None, None
        pt_curr = best_px_curr
        return True

def updateDepthFilter(pt_ref, pt_curr, T_C_R : sp.SE3(),
                    epipolar_direction, depth: cv.Mat, depth_conv2 : cv.Mat):
    
    # triangulation
    TransformationT_Ref_Curr =  T_C_R.inverse()
    point_3d_ref = px2cam(pt_ref) # 3D CORDINATES of refrence point
    point_3d_ref = point_3d_ref / np.sqrt(np.sum(point_3d_ref **2 )) # normalise refence point

    point_3d_curr = px2cam(pt_curr) #3D coordinate current point
    #normalise
    point_3d_curr = point_3d_curr / np.sqrt(np.sum(point_3d_curr **2 ))

    # equation
    # d_ref * f_ref = d_cur * (R_R_C * f_cur) + t_RC
    # d_ref * f_ref = d_cur * f2 + t_RC
    #f2 = (R_R_C * f_cur)
    '''
    Transform into the following matrix equations
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
     A a                 = b
    '''
    trans = TransformationT_Ref_Curr.translation()
    f2 = TransformationT_Ref_Curr.rotationMatrix() @ point_3d_curr
    b = np.array([np.dot(trans, point_3d_ref), np.dot(trans, f2)])
    
    A = np.zeros((2,2))
    A[0, 0] = np.dot(point_3d_ref, point_3d_ref)
    A[0, 1] = -np.dot(-point_3d_ref, f2)
    A[1, 0] = -A[0, 1]
    A[1, 1] = -np.dot(f2,f2)

    ans = A @ b
    xm = ans[0] * point_3d_ref # results in ref
    xn = trans + ans[1] * f2 # results in curr
    p_esti = (xm +xn) /2  # compute avg
    depth_estimation = norm(p_esti) # depth

    # computer variance
    '''
    Calculate uncertainty (error in one pixel)
    a = p - t

    Perturbing p2 by one pixel will cause β to produce a change, 
    which becomes β′. According to the geometric relationship, there are:

    B` = arccos (O2-P2` , -t)
    gamma = π - alpha- β′

    p` = ||t|| * sin(B`) / sin(gamma)
    depth conv = p` - p   p here is also depth estimate

    '''
    
    p = point_3d_ref * depth_estimation
    a = p - trans
    t_norm = norm(trans)
    a_norm = norm(a)
    alpha = np.arccos(np.dot(point_3d_ref, trans)/t_norm)
    beta = np.arccos(np.dot(-a, trans)/ (a_norm * t_norm))

    f_curr_prime = px2cam(pt_curr + epipolar_direction)
    f_curr_prime = f_curr_prime / np.sqrt(np.sum(f_curr_prime **2 ))

    beta_prime = np.arccos(np.dot(f_curr_prime, -trans) / t_norm)

    gamma = np.pi - alpha - beta_prime # gamma = π - alpha- β′
    p_prime = t_norm * np.sin(beta_prime) /np.sin(gamma) #    p` = ||t|| * sin(B`) / sin(gamma)
    d_conv = p_prime - depth_estimation #  depth conv = p` - p   p here is also depth estimate
    d_conv2 = d_conv ** 2

    # Gaussian fusion
    mu = depth[int(pt_ref[1])][int(pt_ref[0])] 
    sigma2 = depth_conv2[int(pt_ref[1])][int(pt_ref[0])] 
    mu_fuse = (d_conv2 * mu + sigma2 * depth_estimation) / (sigma2 + d_conv2)
    sigma_fuse2 = (sigma2 * d_conv2) / (sigma2 + d_conv2)
    depth[int(pt_ref[1])][int(pt_ref[0])]  = mu_fuse
    depth_conv2[int(pt_ref[1])][int(pt_ref[0])]   = sigma_fuse2

    return True , depth, depth_conv2

def update(
    ref_image: cv.Mat,
    curr_image : cv.Mat,
    T_C_R : sp.SE3(), 
    depth : cv.Mat,
    depth_conv2 :cv.Mat):
    '''
    Parameters
        refrence : refrence image
        curr : Current Image
        T_C_R : matrix from refrence to Current
        depth : mean of depth mu
        deptH_conv2 : covariance of depth or std
    '''
    for x in range(BOARDER, WIDTH - BOARDER, 1):
        for y in range(BOARDER, HEIGHT - BOARDER, 1):

            if depth_conv2[y][x] < MIN_CONV or depth_conv2[y][x] > MAX_CONV:
                # algorithm has converged or abort
                continue
            pt_curr = np.array([0,0])
            epipolar_direction = np.zeros([0,0])
            found_good_match : bool = epipolar_search( ref_image=ref_image,\
                                            curr_image=curr_image,\
                                            pt_ref = np.array([x, y]),
                                            T_C_R=T_C_R,\
                                            depth_mu=depth[y][x],
                                            depth_cov=depth_conv2[y][x],
                                            pt_curr=pt_curr,
                                            epipolar_direction=epipolar_direction)



            if not found_good_match:
                continue
            updateDepthFilter(pt_ref = np.array([x,y]), 
                                pt_curr=pt_curr,
                                T_C_R=T_C_R,
                                epipolar_direction = epipolar_direction,
                                depth=depth,
                                depth_conv2=depth_conv2)

def covertQuantToRotatoion( x = 0.0 , y = 0.0 ,z =0.0 , s=0.0 ):
    i_00 = 1 - 2*y*y - 2*z*z
    i_01 = 2*x*y - 2*s*z
    i_02 = (2*x*z) - (2*s*y)

    i_10 = 2*x*y + 2*s*z
    i_11 = 1 - (2*x*x) - (2*z*z)
    i_12 = (2*y*z) - (2*s*x)

    i_20 = (2*x*z) - (2*s*y)
    i_21 = (2*y*z) + (2*s*x)
    i_22 = 1 - (2*x*x) -(2*y*y)
    rotation_matrix = [
        [i_00, i_01, i_02],
        [i_10, i_11, i_12],
        [i_20, i_21, i_22]
        ]
    return np.array(rotation_matrix)

def readData():
    poses = []
    base_dir = os.getcwd()
    lines = []
    frames_dir = base_dir + "/Dense_mapping/test_data/first_200_frames_traj_over_table_input_sequence.txt"
    with open(frames_dir) as file:
            lines = file.readlines()
    poses = []
    for line in lines:
        datxa_ = line.split(" ")
        z = datxa_[ 1 : len(datxa_)]
        tx, ty, tz, qx, qy, qz, qw  = z
        rot_mat = sp.to_orthogonal(covertQuantToRotatoion(float(qx),float(qy),float(qz),float(qw)))
        trans_vec = np.array([float(tx), float(ty), float(tz)])
        pose = sp.SE3(rot_mat, trans_vec)
        poses.append(pose)
    
    camera_images = []
    images_dir = base_dir + "/Dense_mapping/test_data/images/"
    images_files = os.listdir(images_dir)
    for file_name in images_files:
        ac_file_name = images_dir + file_name 
        camera_images.append( ac_file_name )
    
    camera_images.sort()

    depth_dir = base_dir + "/Dense_mapping/test_data/depthmaps/scene_000.depth"
    depth = []
    with open(depth_dir) as file:
            depth = file.readlines()
    depth = depth[0].split(" ")
    depth_ = []
    for v in depth:
        if len(v) != 0:
            convert_ = float(v)
            depth_.append(convert_)
    
    
    idx = 0
    ref_depth = np.zeros((HEIGHT, WIDTH))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ref_depth[y][x] = depth_[idx] / 100
            idx+=1
 
    return poses, camera_images, ref_depth

def evaluateDepth(depth_truth : cv.Mat, depth_estimate : cv.Mat):
    ave_depth_error = 0
    ave_depth_error_sq = 0
    cnt_depth_data = 0
    DM = depth_truth[0].size
    for y  in range(BOARDER, len(depth_truth) - BOARDER, 1):
        for x in range(BOARDER, depth_truth[0].size- BOARDER, 1):
            error = depth_truth[y][x] - depth_estimate[y][x]
            ave_depth_error += error
            ave_depth_error_sq += error * error
            cnt_depth_data +=1
    ave_depth_error /= cnt_depth_data
    ave_depth_error_sq /= cnt_depth_data

    print("Average squared error: ", ave_depth_error_sq,", Average Error: ", ave_depth_error)

def plotDepth(depth_truth : cv.Mat, depth_estimate : cv.Mat):
    cv.imshow("depth Truth", depth_truth * 0.4)
    cv.imshow("depth_etimate", depth_estimate * 0.4)
    cv.imshow("depth_error", np.subtract(depth_truth, depth_estimate))
    cv.waitKey(1)
        
    
if __name__=="__main__":
    poses_TWC, camera_images, ref_depth = readData()
    ref = cv.imread(camera_images[0], 0)
    pose_ref_TWC = poses_TWC[0]
    init_depth = 3.0
    init_conv2 = 3.0
    # intilaise depth maps and conv
    depth = np.full((HEIGHT,WIDTH), init_depth)
    depth_conv2 = np.full((HEIGHT,WIDTH), init_conv2)

    # go through camera images
    for i in range(1, len(camera_images)):
        print("Loop ", i,"*****")
        curr = cv.imread(camera_images[i], 0)
        pose_curr_TWC = poses_TWC[i]
        # coordinate conversion relationship T_C_W * T_W_R = T_C_R
        # 
        poses_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC 
        ret, depth_, depth_conv2_ = update(ref, curr,poses_T_C_R,depth,depth_conv2)
        if ret:
            depth = depth_
            depth_conv2 = depth_conv2_
        evaluateDepth(ref_depth , depth)
        plotDepth(ref_depth, depth)
        cv.imshow("image", curr)
        cv.waitKey(1)
    print("Done")
    base_dir = os.getcwd()
    store_path = base_dir + "/Dense_mapping/depth.png"
    cv.imwrite(store_path, depth)
    

    
from math import floor
import cv2 as cv
from typing import List
import numpy as np

class OpticalFlowTracker():
    def __init__(self, keypoints1 : cv.KeyPoint, keypoints2, success : List[bool], img1 : cv.Mat,
                        img2 : cv.Mat, inverse = True, has_initial = False ) -> None:
     self.keypoints1 = keypoints1
     self.keypoints2 = keypoints2
     self.img1 = img1
     self.img2 = img2
     self.inverse = inverse
     self.success = success
     self.has_initial = has_initial

    def calculateOpticalFlow(self, range_: int):
        half_path_size = 4
        iterations = 10

        for i in range(range_):
            keypoint = self.keypoints1[i]
            dx = 0
            dy = 0
            if (self.has_initial):
                dx = self.keypoints2[i].pt[0] - keypoint[i].pt[0]
                dy = self.keypoints2[i].pt[1] - keypoint[i].pt[1]

            cost = 0
            lastCost = 0
            succ = True # if this point succeed

            # Gauss Newton
            H = np.zeros((2,2))
            b = np.zeros((2,1))
            J = np.array([0, 0]) # jacobian

            for iter in range(iterations):

                if (self.inverse == False):
                    H = np.zeros((2,2))
                    b = np.zeros((2,1))
                else: #only reset b 
                    b = np.zeros((2,1))

                cost = 0

                #cost and jacobian
                # loop through 16 points around key point to get 
                # better estimation of the error or dx
                for x in range(-half_path_size, half_path_size, 1):
                    for y in range(-half_path_size, half_path_size, 1):
                        # error = I_1(x , y ) - I_2(x+dx , y+dy)
                        error = GetPixelValue(self.img1, keypoint.pt[0] + x, keypoint.pt[1] + y) - \
                         GetPixelValue(self.img2, keypoint.pt[0] + x +dx, keypoint.pt[1] + y + dy)
                        # inverse method J is simply gradient of I_2,
                        if (self.inverse == False):
                            # avg between two points equivalent to del I
                            # df/ dh = assume h =2 , f(x + 0.5 *h) - f(x - 0.5 * h)/ h central gradient
                            # df/dh = assume h  = 2 f(x + 1) - f(x - 1)/2
                            # J = - [1/2 * (I_2(x+dx+1, y+dy) - I_2(x+dx -1, y+dy)) , 1/2 * I_2(x+dx, y+dy+1) -I_2(x+dx, y+dy-1)]
                            J = -1.0 * np.array(
                                [0.5 * GetPixelValue(self.img2, keypoint.pt[0] + dx + x + 1, keypoint.pt[1] + dy + y ) - \
                                    GetPixelValue(self.img2, keypoint.pt[0] + dx + x - 1, keypoint.pt[1] + dy + y), \

                                0.5 * GetPixelValue(self.img2, keypoint.pt[0] + dx + x, keypoint.pt[1] + dy + y + 1) - \
                                GetPixelValue(self.img2, keypoint.pt[0] + dx + x, keypoint.pt[1] + dy + y - 1)]
                                )
                        elif (iter == 0):
                            # in inversemode, J keeps same for all iters
                            # this J doesn't change when dx , dy is updateso we can 
                            # store it and only compute error
                            # not sure of proof check paper
                            # -1 * [ 1/2* (I_1(x+1, y) - I_1(x-1, y)) ,  1/2*( I_1(x, y+1) - I_1(x, y-1))]
                            J = -1.0 * np.array(
                                [0.5 * GetPixelValue(self.img1, keypoint.pt[0] + x + 1, keypoint.pt[1] + y ) - \
                                    GetPixelValue(self.img1, keypoint.pt[0] + x - 1, keypoint.pt[1] + y), \

                                0.5 * GetPixelValue(self.img1, keypoint.pt[0] + x, keypoint.pt[1] + dy + y + 1) - \
                                GetPixelValue(self.img1, keypoint.pt[0] + x, keypoint.pt[1] + y - 1)]
                                )
                            
                          

                            b_value = np.dot(-error, J)
                            b_value = b_value.reshape(2,1)
                            b += b_value
                            cost += np.dot(error, error)
                            if (self.inverse == False or iter==0):
                                H += np.dot(J , J.T)
                update = np.linalg.lstsq(H, -b, rcond=None)[0]
                if not update[0]:
                    print(" update is invalid probably H is irreversible")
                    succ = False
                    break
                if iter>0 and cost>lastCost:
                    break
                #update dx , dy
                dx += update[0]
                dy += update[1]
                lastCost = cost
                succ = True
                value_mean = (update).mean(axis=0)
                dx_norm = np.sqrt(abs(value_mean))
                if (dx_norm < 1e-6):
                    break
            self.success[i] = succ

            # set keypoint 2
            self.keypoints2[i].pt[0] = keypoint.pt[0] + dx 
            self.keypoints2[i].pt[1] = keypoint.pt[1] + dy

def GetPixelValue(img : cv.Mat,  x : float,  y : float):
    # set boundary condition for error handling
    if (x < 0):
        x = 0
    if (y < 0):
         y = 0
    if (x >= len(img[0]) - 1):
        x = len(img[0]) - 2
    if (y >= len(img) - 1):
         y = len(img) - 2

    xx = x - floor(x)
    yy = y - floor(y)
    
    x_a1 = min(len(img) - 1, int(x) + 1)
    y_a1 = min(len(img[0]) - 1, int(y) + 1)
    
    return (1 - xx) * (1 - yy) * img[y][x] +\
         xx * (1 - yy) * img[y][x_a1] + (1 - xx) * yy * img[y_a1][x]+\
          xx * yy * img[y_a1][x_a1]


def OpticalFlowSingleLevel(
    img1 : cv.Mat,
    img2 : cv.Mat,
    kp1,
    kp2,
    success : List[bool],
    inverse : bool, 
    has_initial : bool):

    kp2.reshape(kp1.shape)
    success.reshape(kp1.shape)
    tracker = OpticalFlowTracker(img1, img2, kp1, kp2, success, inverse, has_initial)
    # impliment multi processing python here
    # for each range 0 to kp1.size() run calculate optical flow
    # parallel_for_(Range(0, kp1.size()),
    #               std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));


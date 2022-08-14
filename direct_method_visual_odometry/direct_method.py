# compute jacobian in parallel
import cv2 as cv
import sophus as sp
import numpy as np
from threading import Thread, Lock
from typing import List
class JacobianAccumulator():
    def __init__(self, img1 : cv.Mat, img2 : cv.Mat,
                        pixel_refrence : List , depth_refrence : List, T21 : sp.SE3()) -> None:
        self.img1 = img1
        self.img2 = img2
        self.pixel_refrence = pixel_refrence
        self.depth_refrence = depth_refrence
        self.T21 = T21
        self.projection = [[0,0]] * len(self.pixel_refrence)

        # declare Hessian and bias
        self.H = np.zeros((6,6))
        self.b = np.zeros((6,1))

        self.cost = 0
        self.mutex = Lock() # variable to be shared by muliple threads

    
    def getbias(self):
        return self.b

    
    def getHessian(self):
        return self.H

    def getCost(self):
        return self.cost

    def reset(self):
        # reset state of hessian and bias
        self.H = np.zeros((6,6))
        self.b = np.zeros((6,1))

        self.cost = 0

    def getProjectedPoints(self):
        return self.projection

    def accumulate_jacobians(self, range):

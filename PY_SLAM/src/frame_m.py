import sophus as sp
import threading
import cv2 as cv
class FrameM():
    def __init__(self, id = None , time_stamp = None, pose : sp.SE3() = None, 
                    left : cv.Mat = None, right : cv.Mat = None) -> None:
        pass
        self.is_keyFrame = False
        self.timeStamp = time_stamp
        self.pose = pose
        self.pose_mutex_ = threading.Lock()
        
    def getPose(self):
        return self.pose

    def SetPose(self, pose : sp.SE3()):
        self.pose = pose
    def SetKeyFrame():
        return NotImplementedError
    def CreateFrame():
        return NotImplementedError

    
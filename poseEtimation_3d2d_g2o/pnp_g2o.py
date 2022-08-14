from unicodedata import name
from scipy.linalg import lstsq
import sys
import os
import cv2 as cv
import numpy as np
import g2o
import sophus as sp

base_dir = os.getcwd()
folder_dir = "/poseEstimation_3d2d_bundleAdjustment"
sys.path.insert(0, base_dir)
from utils import convert_pixel_to_cam as p2c
from utils import feature_matches as fm



class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()





if __name__ == '__main__':
   
    # image1 = os.path.join(base_dir, folder_dir + "/images/1.png")
    image1_path = base_dir + folder_dir + "/images/1.png"
    image2_path = base_dir + folder_dir + "/images/2.png"
    image3_path = base_dir + folder_dir + "/images/1_depth.png"

    # image2 = os.path.join(base_dir, folder_dir + "/images/2.png")

    #depth images
    # image3 = os.path.join(base_dir, folder_dir + "/images/1_depth.png")


    image1 = cv.imread(image1_path, cv.IMREAD_COLOR )
    image2 = cv.imread(image2_path, cv.IMREAD_COLOR )

  
        

    keypoints_1 = []
    keypoints_2 = []
    matches = []
    matches , keypoints_1, keypoints_2 = fm.find_feature_matches(image1, image2)


    depth_image = cv.imread(image3_path, cv.IMREAD_UNCHANGED)
    print("depth image shape", depth_image.shape)
    K = [[ 520.9, 0, 325.1], \
        [0, 521.0, 249.7],\
        [ 0, 0, 1]]
    pts_3d = []
    pts_2d = []

    for match in matches:
        idx = int(keypoints_1[match.queryIdx].pt[0])
        idy = int(keypoints_1[match.queryIdx].pt[1])
        d = depth_image[idy][idx]
        if d ==0:
            continue
        dd = d/5000.0
        p1 = p2c.pixel_to_cam(keypoints_1[match.queryIdx].pt, K)

        pos_3d = [p1[0] * dd, p1[1] * dd,  dd]
        pts_3d.append(pos_3d)

        pts_2d.append(keypoints_2[match.trainIdx].pt)
    print("number of 3d-2d pairs", len(pts_3d), len(pts_2d))

    distortion_coeffs = np.zeros((4,1))
    pts_3d_np = np.array(pts_3d)
    pts_2d_np = np.array(pts_2d)
    K_np = np.array(K)


    success, vector_rotation, vector_translation = cv.solvePnP(pts_3d_np, pts_2d_np, K_np,distortion_coeffs, flags=0,useExtrinsicGuess=False)
    print("rotation matrix", vector_rotation)
    print("translation vector",vector_translation )
    print(vector_rotation.shape)
   
    R = cv.Rodrigues(vector_rotation)
  
    print("R from rodrigues", R)

    # coputing bundle adjustent
    BundleAdjustment(pts_3d_np, pts_2d_np, K)
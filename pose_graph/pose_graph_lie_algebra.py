import re
import sophus as sp
import os

import numpy as np
from typing import List
import numpy
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=30):
        super().initialize_optimization()
        super().set_verbose(True)
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


def matrix_exp(rotation, translation):
    return np.array([
        [rotation[0][0], rotation[0][1], rotation[0][2],translation[0] ],
        [rotation[1][0], rotation[1][1], rotation[1][2],translation[1]],
        [rotation[2][0],rotation[2][1],rotation[2][2], translation[2]],
        [0,0,0,1]
    ])

# converts Quant to rotation matrix  s is the real part and well x y z are what they are
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

def createPoseRotationMatTranslation(rotaion_matrix , translation_matrix):
    # t_mat = matrix_exp(sp.to_orthogonal(rotaion_matrix) , translation_matrix)
    return sp.SE3(sp.to_orthogonal(rotaion_matrix) , translation_matrix) # R, t

def createInformationMatrix(main_diag = [10000,10000,10000,40000,40000,40000]):
    information_matrix = np.diag(main_diag)
    return information_matrix



class Vertex():
    def __init__(self, id, tx, ty, tz, qx,qy,qz,qw) -> None:
        self.id = id
        self.tx = tx 
        self.ty = ty
        self.tz = tz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

    def getId(self):
        return self.id

    def getRotationMatrix(self):
        return covertQuantToRotatoion( x= self.qx, y = self.qy , z = self.qz, s = self.qw)

    def getTranslationVector(self):
        return np.array([self.tx, self.ty, self.tz])

    def getTPose(self):
        rotation_mat = sp.to_orthogonal(self.getRotationMatrix())
        trans_vec = self.getTranslationVector()
        # translation_matrix = createPoseRotationMatTranslation(self.getRotationMatrix(),np.ones(3))
        # translation_matrix = g2o.Isometry3d(rotation_mat, trans_vec)

        q = np.array([self.qw, self.qx, self.qy ,self.qz])
        q = g2o.Quaternion(q)
        translation_matrix = g2o.Isometry3d(q , trans_vec) 
        return translation_matrix
    
        

class Edges:
    def __init__(self, id0, id1, tx, ty, tz, qx, qy, qz, qw, information_mat ) -> None:
        self.id0 = id0
        self.id1 = id1
        self.tx = tx 
        self.ty = ty
        self.tz = tz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.information_mat = information_mat

    def getIDS(self):

        return self.id0 , self.id1

    def getRotationMatrix(self):

        return covertQuantToRotatoion(x= self.qx, y = self.qy , z = self.qz, s = self.qw)

    def getTranslationVector(self):

        return np.array([self.tx, self.ty, self.tz])

    def getInformationMatrix(self):
        return createInformationMatrix()

    def getTPose(self):
        # translation_matrix =  createPoseRotationMatTranslation(self.getRotationMatrix(), self.getTranslationVector())
        # rotation_mat = self.getRotationMatrix()
        trans_vec = self.getTranslationVector()
        # translation_matrix = g2o.Isometry3d(rotation_mat, trans_vec)
        q = np.array([self.qw, self.qx, self.qy ,self.qz,])
        q = g2o.Quaternion(q)
        translation_matrix = g2o.Isometry3d(q , trans_vec) 
        return translation_matrix

base_dir = os.getcwd()
folder_dir = "pose_graph/data.txt"


if __name__ == '__main__':
    lines = []
    data_file_path = os.path.join(base_dir, folder_dir)
    with open(data_file_path) as file:
            lines = file.readlines()
    print("number of data points: " , len(lines))

    # vertices have format VERTEX_SE3:QUAT 1 -0.250786 -0.0328449 99.981 0.705413 0.0432253 0.705946 -0.0465295 
    # we use first letter V to categorise and E for edges




    vertices : List[Vertex] = list()
    edges : List[Edges] = list()

    for data_item in lines:
        # using first letter of string
        if data_item[0] == "V": # vertex

            #seperate words and values
            data_values = data_item.split(" ")
            # slicing to access values
            data_values = data_values[1:len(data_values)-1]
            id, tx, ty, tz, qx, qy, qz, qw  = data_values
            vetrex_node = Vertex(int(id), float(tx), float(ty), float(tz), float(qx), float(qy), float(qz),float(qw) )
            vertices.append(vetrex_node)
        else:# edges
            data_values = data_item.split(" ")
            # slicing to access values
            data_values = data_values[1:len(data_values)-1]
            
            id0, id1, tx, ty, tz, qx, qy, qz, qw, *information_matrix = data_values
            edge_node = Edges(int(id0), int(id1), float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw), information_matrix)
            edges.append(edge_node)

    print("Number of vertex node", len(vertices))
    print("number of edges", len(edges))

    pose_graph_optimiser = PoseGraphOptimization()

    for vertex in  vertices:
        id = vertex.getId()
        translation_pose =  vertex.getTPose()
        pose_graph_optimiser.add_vertex(vertex.getId(), vertex.getTPose() )

    for edge in edges:
        pose_graph_optimiser.add_edge(vertices=edge.getIDS() ,measurement=edge.getTPose(), information=edge.getInformationMatrix(),robust_kernel = g2o.RobustKernelHuber() )

    # pose_graph_optimiser.optimize()
    x = vertices[0].getTPose()
    print(vertices[0].getTPose().R)

   
    
    # now using the 
    input_file = "/Users/emma/dev/visual-slam-python/pose_graph/sphere.g2o"
    out_put_file =""
    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)

    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(True)
    optimizer.set_algorithm(solver)

    optimizer.load(input_file)
   

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()), end='\n\n')

    optimizer.initialize_optimization()
    # optimizer.optimize(29)

    # if len(out_put_file) > 0:
    optimizer.save("/Users/emma/dev/visual-slam-python/pose_graph/result.g2o")

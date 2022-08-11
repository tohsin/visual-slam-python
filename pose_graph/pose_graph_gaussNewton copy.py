import numpy as np
import sophus as sp
from typing import List
import os
import g2o
import scipy.sparse
import scipy.sparse.linalg
from scipy.spatial.transform import Rotation as R

def compute_jrInv(e : sp.SO3()):
    I = np.identity(6)
    # J =  np.zeros((6, 6))
    
    J = np.block([
        [sp.SO3.hat(e.so3().log()) , sp.SO3.hat(e.translation())],
        [np.zeros((3,3)), sp.SO3.hat(e.so3().log())]]
        )
    # print(J.shape)
    J = (0.5 * J ) + np.identity(6)
     #SO3d::hat(e.so3().log());
    return J
    
def compute_adj_lie_algebra(T : sp.SE3()):
    R_ = T.rotationMatrix()
    t_ = T.translation() 
    adj_ = np.block(
        [
            [R_ , np.cross(t_,R_)],
            [np.zeros((3,3)), R_]
        ]
    )
    return adj_

def compute_error(pose_Ti, pose_Tj , measurment_Ti_Tj):
    '''
     error = measurement  - Ti.inverse() Tj
     ln(measurement.inverse() * Ti.inverse() * tj) convert to vector symbol

    '''

    error = ( measurment_Ti_Tj.inverse()  * ( pose_Tj.inverse() * pose_Ti ) )
    error = error.log()
    return error
    # inverse of matrix
    # y = np.linalg.inv(x) 
def compute_jacobian(error, pose_Tj):
    J = compute_jrInv( sp.SE3.exp(error))
    mat_inv = pose_Tj.inverse()
    adj_ = compute_adj_lie_algebra(T = mat_inv)
    _jacobianOplusXi = -J @ adj_
    _jacobianOplusXj = J @ adj_
    return _jacobianOplusXi, _jacobianOplusXj


def update_pose(pose , dx):
    return sp.SE3.exp(dx) * pose

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


def id2index(value_id, num_params):
    return slice((num_params * value_id), (num_params * (value_id + 1)))

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
        self.pose = self.getTPose()

    def getId(self):
        return self.id

    def getRotationMatrix(self):
        
        return covertQuantToRotatoion( x= self.qx, y = self.qy , z = self.qz, s = self.qw)

    def getTranslationVector(self):
        return np.array([self.tx, self.ty, self.tz])

    def getTPose(self):
        rotation_mat = sp.to_orthogonal(self.getRotationMatrix())
        trans_vec = self.getTranslationVector()
        translation_matrix = sp.SE3(rotation_mat, trans_vec)
        return translation_matrix
    
        
class Edges:
    def __init__(self, id0, id1, tx, ty, tz, qx, qy, qz, qw, information_mat ) -> None:
        self.id_i = id0
        self.id_j = id1
        self.tx = tx 
        self.ty = ty
        self.tz = tz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.information_mat = self.getInformationMatrix()
        self.measurement = self.getTPose()
    def getIDS(self):

        return self.id_i , self.id_j

    def getRotationMatrix(self):

        return covertQuantToRotatoion(x= self.qx, y = self.qy , z = self.qz, s = self.qw)

    def getTranslationVector(self):

        return np.array([self.tx, self.ty, self.tz])

    def getInformationMatrix(self):
        
        return createInformationMatrix()

    def getTPose(self):
        # translation_matrix =  createPoseRotationMatTranslation(self.getRotationMatrix(), self.getTranslationVector())
        rotation_mat = sp.to_orthogonal(self.getRotationMatrix())
        trans_vec = self.getTranslationVector()
        translation_matrix = sp.SE3(rotation_mat, trans_vec)
        return translation_matrix


def GaussNetwonPoseGraph(vertices : List[Vertex] = list() , edges : List[Edges] = list()):
    for _ in range(15):
        num_nodes = len(vertices)
        num_params = 6

                # define the sparce matrix needed for big H
                #solve H * delx = -b 
                
        dim_v = num_nodes * 6
        H = scipy.sparse.csc_matrix((dim_v ,dim_v))
        b = scipy.sparse.csc_matrix((dim_v, 1))
        cost = 0
    
        for edge in edges:
            index_i , index_j = edge.getIDS()
            vertex_i , vertex_j = vertices[index_i] , vertices[index_j]
            measurment_T_ij = edge.measurement
            error = compute_error(pose_Ti = vertex_i.pose, pose_Tj = vertex_j.pose , measurment_Ti_Tj = measurment_T_ij)
            # consider impliment robust kernel here
         
            
            jacobian_xi, jacobian_xj =  compute_jacobian(error=error, pose_Tj=vertex_j.pose)    
            omega = edge.information_mat
            cost += error.T @ omega  @ error
            H_ii = jacobian_xi.T @ omega @ jacobian_xi
            H_ij = jacobian_xi.T @ omega @ jacobian_xj
            H_jj = jacobian_xj.T @ omega @ jacobian_xj

            b_i = -jacobian_xi.T @ omega @ error
            b_j = -jacobian_xj.T @ omega @ error


            H[id2index(index_i, num_params), id2index(index_i, num_params)] += H_ii
            H[id2index(index_i, num_params), id2index(index_j, num_params)] += H_ij
            H[id2index(index_j, num_params), id2index(index_i, num_params)] += H_ij.T
            H[id2index(index_j, num_params), id2index(index_j, num_params)] += H_jj

            b[id2index(index_i,num_params)] += b_i.reshape(6,1)
            b[id2index(index_j,num_params)] += b_j.reshape(6,1)
        
            

        H[:num_params, :num_params] += np.eye(6) # fix the first node
        dx = scipy.sparse.linalg.spsolve(H ,b) # solver to solve linear equation
        dx[np.isnan(dx)] = 0 # remove very small values or none
        dx = np.reshape(dx, (len(vertices), num_params))

        # update poses
        dx_change = 0
        for value_id, update in enumerate(dx):
         
                # dx_change_value = (update).mean(axis=0)
                # dx_change += abs(dx_change_value)
                old_pose = vertices[value_id].pose
                new_pose = update_pose(pose = old_pose, dx = update)
                vertices[value_id].pose = new_pose
            

        print(0.5*cost)
        # check if we converged
        # print("update_mean", dx_change)
        # print("mean error", mean_error)
    
    #compute error


        



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




    # vertices : List[Vertex] = list()
    vertices = [None] * 2500 # 2500 nodes
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
            vertices[int(id)] = vetrex_node
            # vertices.append(vetrex_node)
        else:# edges
            data_values = data_item.split(" ")
            # slicing to access values
            data_values = data_values[1:len(data_values)-1]
            
            id0, id1, tx, ty, tz, qx, qy, qz, qw, *information_matrix = data_values
            edge_node = Edges(int(id0), int(id1), float(tx), float(ty), float(tz), float(qx), float(qy), float(qz), float(qw), information_matrix)
            edges.append(edge_node)

    
    print("Number of vertex node", len(vertices))
    print("number of edges", len(edges))

    # data generation process done

    GaussNetwonPoseGraph(vertices = vertices, edges = edges)
   


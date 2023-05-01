
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)


    def prediction(self, u):
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################

        adjX = np.hstack((self.mu[0:2, 0:2], np.array([[self.mu[1, 2]], [-self.mu[0, 2]]])))
        adjX = np.vstack((adjX, np.array([0, 0, 1])))

        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.propagation(u_se2, adjX)

    def propagation(self, u, adjX):
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################





        self.X_pred = self.mu @ expm(u)

        self.P_pred = self.Sigma + (adjX @ self.W @ adjX.T)
        



        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
       
    def correction(self, Y1, Y2, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        # %             obj.mu = obj.mu_pred;

        H1 = np.array([[-1, 0, landmark1.getPosition()[1]],
                       [0, -1, -landmark1.getPosition()[0]]])

        H2 = np.array([[-1, 0, landmark2.getPosition()[1]],
                       [0, -1, -landmark2.getPosition()[0]]])

        H = np.vstack((H1,H2))

        N = self.X_pred * block_diag(self.V, 0) * np.transpose(self.X_pred)
        N = block_diag(N[0:2, 0:2], N[0:2, 0:2])  # 4 x 4 block-diagonal matrix

        S = np.dot(np.dot(H, self.P_pred), H.T) + N
        L = np.dot(np.dot(self.P_pred, H.T), np.linalg.inv(S))
        b1 = np.hstack((landmark1.getPosition()[0], landmark1.getPosition()[1], 1))
        b2 = np.hstack((landmark2.getPosition()[0], landmark2.getPosition()[1], 1))


        nu = np.dot(block_diag(self.X_pred, self.X_pred), np.vstack((Y1.reshape(-1,1), Y2.reshape(-1,1))) ) - np.vstack((b1.reshape(-1, 1), b2.reshape(-1, 1)))
        nu = np.hstack((nu[0:2, 0], nu[3:5, 0]))

        twist = np.dot(L, nu)
        twist_hat = np.array([[0, -twist[2], twist[0]],
                              [twist[2], 0, twist[1]],
                              [0, 0, 0]])

  
   
        X = np.dot(expm(twist_hat), self.X_pred)

        I = np.eye(3)
        xx = X[0,2]
        yy = X[1,2]
        angle = np.arctan2(X[1,0],X[0,0])
        X = np.array([xx,yy,angle])

        tempor = I - np.dot(L,H)
        self.Sigma = np.dot(np.dot(tempor, self.P_pred), tempor.T) + np.dot(np.dot(L, N), L.T)
        self.mu = self.pose_mat(X)


        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H

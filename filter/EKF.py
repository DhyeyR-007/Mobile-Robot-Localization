import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u):

        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################


        G_t = self.Gfun(X,u)

        # G_t = wrap2Pi(G_t)
       
        
        V_t = self.Vfun(X,u)
        M_t = self.M(u)



        X_pred = self.gfun(X,u)
        # # X_pred[0] = wrap2Pi(X_pred[0])
        # # X_pred[1] = wrap2Pi(X_pred[1])
        X_pred[2] = wrap2Pi(X_pred[2])

        
        P_pred = (G_t @ P @ G_t.transpose()) + (V_t @ M_t @ V_t.transpose())



        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)


    def correction(self, z, landmarks):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = self.state_.getState()
        P_predict = self.state_.getCovariance()
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        



                
        y_k1 = z[0:2] - self.hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict)
        y_k1[0] = wrap2Pi(y_k1[0])

        y_k2 = z[3:5] - self.hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict)
        y_k2[0] = wrap2Pi(y_k2[0])


        H_t = np.vstack((self.Hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict,z[0:2]), self.Hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict,z[3:5])))


        Q_t = np.kron(np.eye(2),self.Q[0:2,0:2])
        

        S_t = np.dot(np.dot(H_t ,P_predict) , H_t.transpose()) + Q_t


        
        K_t = np.dot(np.dot(P_predict, H_t.transpose()), np.linalg.inv(S_t))
        

        
        temporar = np.dot(K_t, np.transpose(np.hstack((y_k1,y_k2))))
        X = X_predict + temporar
        X[2] = wrap2Pi(X[2])

        I = np.eye(3)
        P = np.dot((I - np.dot(K_t, H_t)), P_predict)
        
        



        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
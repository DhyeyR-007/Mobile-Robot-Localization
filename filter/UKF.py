
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi


class UKF:
    # UKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance
        
        self.kappa_g = init.kappa_g
        
        self.state_ = RobotState()
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)



    def prediction(self, u):
        # prior belief
        X = self.state_.getState()
        P = self.state_.getCovariance()

        ###############################################################################
        # TODO: Implement the prediction step for UKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        

        M_t = self.M(u)
        Q_t = self.Q



        mu_a_t__1 = np.concatenate((X.transpose(), np.zeros(3).transpose(), np.zeros(2).transpose())).reshape(-1,1)        
        sigma_a_t__1 = block_diag(P, M_t, Q_t)
        self.sigma_point(mu_a_t__1 , sigma_a_t__1, self.kappa_g)



        self.X_dash_x_t = np.zeros_like(self.X[0:3,:])
        for i in range(len(self.X[0])):
            self.X_dash_x_t[:,i] = self.gfun(self.X[0:3,i],u + self.X[3:6,i])
        
        
        mew_dash_t = np.zeros_like(self.X_dash_x_t[:,0])
        for j in range(len(self.X[0])):
            mew_dash_t += self.w[j] * self.X_dash_x_t[:,j]

        mew_dash_t[2] = wrap2Pi(mew_dash_t[2])


        sigma_dash_t = np.zeros((3,3))
        for k in range(len(self.X[0])):
            mat = self.X_dash_x_t[:,k] - mew_dash_t
            mat[2] = wrap2Pi(mat[2])
            placeholder = self.w[k] * mat.reshape(3,1) @ mat.reshape(1,3)
            sigma_dash_t += placeholder

        X_pred =  mew_dash_t 

        P_pred =  sigma_dash_t + M_t
            
                                                      
    
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################


        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)

    def correction(self, z, landmarks):

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


        Z_t = np.zeros((4,len(self.X[0])))

        for i in range(len(self.X[0])):

            Z_t[0:2,i] = self.hfun(landmark1.getPosition()[0], landmark1.getPosition()[1], self.X_dash_x_t[:,i]) + self.X[6:,i]
            Z_t[2:,i] = self.hfun(landmark2.getPosition()[0], landmark2.getPosition()[1], self.X_dash_x_t[:,i]) + self.X[6:,i]

            Z_t[0] = wrap2Pi(Z_t[0])
            Z_t[2] = wrap2Pi(Z_t[2])


        
        z_t = np.zeros((len(self.w[0] * Z_t[:,0])))
        for i in range(len(self.X[0])):
            z_t += self.w[i] * Z_t[:,i]
            
        z_t[0] = wrap2Pi(z_t[0])
        z_t[2] = wrap2Pi(z_t[2])





        placeholder_0 = self.w[0] * (Z_t[:,0] - z_t).reshape(4,1) @ (Z_t[:,0] - z_t).reshape(1,4)
        S_t = np.zeros(placeholder_0.shape)
        for i in range(len(self.X[0])):
           S_t += self.w[i] * (Z_t[:,i] - z_t).reshape(4,1) @ (Z_t[:,i] - z_t).reshape(1,4)
        S_t += np.kron(np.eye(2),self.Q[0:2,0:2])

        

        placeholder_1= self.w[0] * (self.X_dash_x_t[:,0] - X_predict).reshape(3,1) @ (Z_t[:,0] - z_t).reshape(1,4)
        sigma_xz_t = np.zeros(placeholder_1.shape)
        for i in range(len(self.X[0])):
            sigma_xz_t += self.w[i] * (self.X_dash_x_t[:,i] - X_predict).reshape(3,1) @ (Z_t[:,i] - z_t).reshape(1,4)
        




        K_t = np.dot(sigma_xz_t,np.linalg.inv(S_t))
        z_t_without_hat = np.hstack((z[0:2],z[3:5]))
        delta_t = (z_t_without_hat - z_t).reshape(4,1)

        mew____t = X_predict.reshape(3,1) + (K_t @ delta_t).reshape(3,1)
        mew____t[2] = wrap2Pi(mew____t[2])
        X = mew____t.reshape(3,)

        sigma__t = P_predict - (K_t @ S_t @ np.transpose(K_t))
        P = sigma__t






        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(P)

    def sigma_point(self, mean, cov, kappa):
        self.n = len(mean) # dim of state
        L = np.sqrt(self.n + kappa) * np.linalg.cholesky(cov)
        Y = mean.repeat(len(mean), axis=1)
        self.X = np.hstack((mean, Y+L, Y-L))
        self.w = np.zeros([2 * self.n + 1, 1])
        self.w[0] = kappa / (self.n + kappa)
        self.w[1:] = 1 / (2 * (self.n + kappa))
        self.w = self.w.reshape(-1)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state
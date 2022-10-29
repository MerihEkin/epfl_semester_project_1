
from cmath import sin, cos
from math import gamma, pi, sqrt, exp, pow
from tkinter import W
import numpy as np
import numpy.linalg as LA
import numpy.matlib as matlib
from enum import Enum

from vartools.dynamical_systems._base import DynamicalSystem
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.avoidance.modulation import obs_avoidance_interpolation_moving


class Algorithm(Enum):
    Plain = 1
    Weighted1 = 2
    Weighted2 = 3
    DeltaVelocity = 4


class GmmVariables:
    def __init__(self, mu, priors, sigma) -> None:
        self.mu = mu  # np.array(mu)
        self.priors = priors  # np.array(priors)
        self.sigma = sigma  # np.array(sigma)


class LpvDs(DynamicalSystem):
    def __init__(self, A_k, b_k, eps, realmin, obstacle_environment: ObstacleContainer = ObstacleContainer(),
                 target: np.array = np.array([0.0, 0.0]), algorithm: Algorithm = Algorithm.Plain,):
        self.eps = eps
        self.realmin = realmin
        self.A_k = A_k
        self.b_k = b_k
        self.algorithm = algorithm
        self.target = target
        self.obstacle_environment = obstacle_environment

    def evaluate(self, x, ds_gmm: GmmVariables):
        # Auxiliary Variables
        N, M = x.shape
        K = ds_gmm.priors.shape[1]  # K = length(ds_gmm.Priors);

        # Posterior Probabilities per local DS
        beta_k_x = self.posterior_probs_gmm(x, ds_gmm, 'norm')

        # Output Velocity
        x_dot = np.zeros((N, M))  # x_dot = zeros(N,M);
        for i in range(M):
            if self.b_k.shape[1] > 1:
                f_g = np.zeros((N, K))
                for k in range(K):
                    f_g[:, k] = beta_k_x[k, i] * \
                        (self.A_k[:, :, k] @ x[:, i] + self.b_k[:, k])
                f_g = np.sum(f_g, 1)
            else:
                f_g = self.A_k*x[:, i] + self.b_k
            x_dot[:, i] = f_g

        if self.algorithm == Algorithm.Plain:
            return x_dot
        elif self.algorithm == Algorithm.Weighted1:
            vel, alpha = self.compute_alpha(x)
            x_dot.shape = (2,)
            return (1 - alpha) * x_dot + alpha * vel
        elif self.algorithm == Algorithm.Weighted2:
            vel, alpha = self.compute_beta(x, x_dot)
            x_dot.shape = (2,)
            return (1 - alpha) * x_dot + alpha * vel
        elif self.algorithm == Algorithm.DeltaVelocity:
            pass

    def compute_alpha(self, position):
        position.shape = (2,)

        N_obs = len(self.obstacle_environment._obstacle_list)
        if not N_obs:
            return np.array([0, 0]), 0

        Gamma = np.zeros((N_obs))

        for n in range(N_obs):
            Gamma[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)

        dirs = np.zeros((2, N_obs))
        for n in range(N_obs):
            ps = self.obstacle_environment._obstacle_list[n].pose.transform_position_from_reference_to_local(
                position)
            a = self.obstacle_environment._obstacle_list[n].semiaxes[0]
            b = self.obstacle_environment._obstacle_list[n].semiaxes[1]
            theta = self.obstacle_environment._obstacle_list[n].orientation_in_degree * np.pi / 180
            a_ = a*cos(theta) + b*sin(theta)
            b_ = a*sin(theta) + b*cos(theta)
            sgn = 1 if ps[0] > 0 else -1
            dirs[0, n] = sgn*a_ - ps[0] if abs(ps[0]) > a_ else 0
            sgn = 1 if ps[1] > 0 else -1
            dirs[1, n] = sgn*b_ - ps[1] if abs(ps[1]) > b_ else 0
            dir_norm = np.linalg.norm(dirs[:, n], 2)
            dirs[:, n] = dirs[:, n] / dir_norm if dir_norm > 0 else dirs[:, n]
            print(dirs[:, n])

        max_gamma = np.max(1/Gamma)

        alpha = 1 / (1 + exp(-100 * max_gamma + 50))

        vel = self.target - position
        vel /= np.linalg.norm(vel, 2)
        vel = obs_avoidance_interpolation_moving(
            position, vel, self.obstacle_environment
        )
        return vel, alpha

    def compute_beta(self, position, velocity):
        position.shape = (2,)
        velocity.shape = (2,)

        N_obs = len(self.obstacle_environment._obstacle_list)
        if not N_obs:
            return np.array([0, 0]), 0

        Gamma = np.zeros((N_obs))

        for n in range(N_obs):
            Gamma[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)

        velocity /= np.linalg.norm(velocity, 2)

        weights = np.zeros((N_obs))
        direction = np.zeros((2))
        directions = np.zeros((2, N_obs))
        for n in range(N_obs):
            ps = self.obstacle_environment._obstacle_list[n].pose.transform_position_from_reference_to_local(
                position)
            a = self.obstacle_environment._obstacle_list[n].semiaxes[0]
            b = self.obstacle_environment._obstacle_list[n].semiaxes[1]
            theta = self.obstacle_environment._obstacle_list[n].orientation_in_degree * np.pi / 180
            a_ = a*cos(theta) + b*sin(theta)
            b_ = a*sin(theta) + b*cos(theta)
            sgn = 1 if ps[0] > 0 else -1
            direction[0] = sgn*a_ - ps[0] if abs(ps[0]) > a_ else 0
            sgn = 1 if ps[1] > 0 else -1
            direction[1] = sgn*b_ - ps[1] if abs(ps[1]) > b_ else 0
            dir_norm = np.linalg.norm(direction, 2)
            direction = direction / dir_norm if dir_norm > 0 else direction
            weights[n] = direction[0] * velocity[0] + \
                direction[1] * velocity[1]
            weights[n] = 1 if weights[n] > 0 else 0
            directions[:, n] = direction

        Gamma = 1/Gamma
        max_gamma = np.max(Gamma)
        index = np.argmax(Gamma)
        vel = self.target - position
        x_sign = -1 if vel[0] < 0 else 1
        y_sign = -1 if vel[1] < 0 else 1
        ref_direction = np.array(
            [x_sign*abs(directions[1, index]), y_sign*abs(directions[0, index])])

        alpha = 1 / (1 + exp(-100 * max_gamma + 50))

        vel = obs_avoidance_interpolation_moving(
            position, ref_direction, self.obstacle_environment
        )
        return vel, alpha

    def posterior_probs_gmm(self, x, ds_gmm: GmmVariables, type):
        N, M = x.shape
        K = ds_gmm.priors.shape[1]
        Mu = ds_gmm.mu
        Priors = ds_gmm.priors
        Sigma = ds_gmm.sigma
        Px_k = np.zeros((K, M))

        for k in range(K):
            Px_k[k, :] = self.ml_gaussian_pdf(
                x, Mu[:, k], Sigma[:, :, k]) + self.eps

        alpha_Px_k = np.multiply(matlib.repmat(
            np.transpose(Priors), 1, M), Px_k)

        if type == 'norm':
            Px_k = np.divide(alpha_Px_k, matlib.repmat(
                np.sum(alpha_Px_k, 0), K, 1))
        elif type == 'un-norm':
            Px_k = alpha_Px_k

        return Px_k

    def ml_gaussian_pdf(self, Data, Mu, Sigma):
        if Data.shape[0] == 1:
            Data.shape = (Data.size, 1)
        if Mu.shape[0] == 1:
            Mu.shape = (Mu.size, 1)

        nbVar, nbData = Data.shape
        Mus = matlib.repmat(Mu, 1, nbData)
        Mus.shape = (Mu.shape[0], nbData)
        Data = np.transpose(Data - Mus)
        prob = np.sum(np.multiply(np.matmul(Data, LA.inv(Sigma)), Data), 1)
        prob = exp(-0.5 * prob) / sqrt(pow((2*pi), nbVar)) * \
            (abs(LA.det(Sigma) + self.realmin)) + self.realmin
        return prob

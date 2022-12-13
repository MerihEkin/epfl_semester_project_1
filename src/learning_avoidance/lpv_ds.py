
from cmath import sin, cos
from math import gamma, pi, sqrt, exp, pow
import numpy as np
import numpy.linalg as LA
import numpy.matlib as matlib
from enum import Enum

from vartools.dynamical_systems._base import DynamicalSystem


class GmmVariables:
    def __init__(self, mu, priors, sigma) -> None:
        self.mu = mu
        self.priors = priors
        self.sigma = sigma


class LpvDs(DynamicalSystem):
    def __init__(self, A_k, b_k, ds_gmm: GmmVariables, eps=1e-5, realmin=2e-100):
        self.A_k = A_k
        self.b_k = b_k
        self.ds_gmm = ds_gmm
        self.eps = eps
        self.realmin = realmin

    def evaluate(self, x):
        # Auxiliary Variables
        x.shape = (x.size, 1)
        N, M = x.shape
        K = self.ds_gmm.priors.shape[1]  # K = length(ds_gmm.Priors);

        # Posterior Probabilities per local DS
        beta_k_x = self.posterior_probs_gmm(x, 'norm')

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
        x.shape = (x.size,)
        x_dot.shape = (x.size,)
        return x_dot

    def posterior_probs_gmm(self, x, type):
        N, M = x.shape
        K = self.ds_gmm.priors.shape[1]
        Mu = self.ds_gmm.mu
        Priors = self.ds_gmm.priors
        Sigma = self.ds_gmm.sigma
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
        Data = Data.copy()
        Mu = Mu.copy()
        Sigma = Sigma.copy()
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

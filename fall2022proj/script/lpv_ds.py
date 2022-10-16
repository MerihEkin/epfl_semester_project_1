
from math import pi, sqrt, exp, pow
import numpy as np
import numpy.linalg as LA
import numpy.matlib as matlib


class GmmVariables:
    def __init__(self, mu, priors, sigma) -> None:
        self.mu = mu  # np.array(mu)
        self.priors = priors  # np.array(priors)
        self.sigma = sigma  # np.array(sigma)


class LpvDs:
    def __init__(self, eps, realmin):
        self.eps = eps
        self.realmin = realmin

    def evaluate(self, x, ds_gmm: GmmVariables, A_g, b_g):
        # Auxiliary Variables
        N, M = x.shape
        K = ds_gmm.priors.shape[1]  # K = length(ds_gmm.Priors);

        # Posterior Probabilities per local DS
        beta_k_x = self.posterior_probs_gmm(x, ds_gmm, 'norm')

        # Output Velocity
        x_dot = np.zeros((N, M))  # x_dot = zeros(N,M);
        for i in range(M):
            if b_g.shape[1] > 1:
                f_g = np.zeros((N, K))
                for k in range(K):
                    f_g[:, k] = beta_k_x[k, i] * \
                        (A_g[:, :, k] @ x[:, i] + b_g[:, k])
                f_g = np.sum(f_g, 1)
            else:
                f_g = A_g*x[:, i] + b_g
            x_dot[:, i] = f_g
        return x_dot

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

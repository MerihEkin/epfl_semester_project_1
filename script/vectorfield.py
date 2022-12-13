from re import L
import time
import lpv_ds
import matlab.engine
import numpy as np
import random

import pyLasaDataset as lasa
import matplotlib as mpl
import matplotlib.pyplot as plt

eps = 1e-5
realmin = 2e-100


def get_gmm_from_matlab(position, velocity, final_pos: np.array):

    lyap_constr = 2
    init_cvx = 1
    symm_constr = 0

    est_options = {
        "type": 0,
        "maxK": 15.0,
        "fixed_K": matlab.double([]),
        "samplerIter": 20.0,
        "do_plots": 0,
        "sub_sample": 1,
        "estimate_l": 1.0,
        "l_sensitivity": 2.0,
        "length_scale": matlab.double([]),
    }

    data_py = np.vstack((position, velocity))

    if not "matlab_eng" in locals():
        matlab_eng = matlab.engine.start_matlab()

    pos_array = matlab.double(position.astype('float64'))
    vel_array = matlab.double(velocity.astype('float64'))
    Data = matlab.double(data_py.astype('float64'))

    priors, mu, sigma = matlab_eng.fit_gmm(
        pos_array, vel_array, est_options, nargout=3)

    ds_gmm = {
        "Mu": mu,
        "Priors": priors,
        "Sigma": sigma,
    }

    Vxf = matlab_eng.learn_wsaqf(Data, nargout=1)
    P_opt = Vxf["P"]

    att = matlab.double(final_pos.astype('float64'))
    att = matlab_eng.transpose(att)

    A_k, b_k, P = matlab_eng.optimize_lpv_ds_from_data(
        Data, att, lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr, nargout=3)

    matlab_eng.quit()

    priors = np.array(priors)
    mu = np.array(mu)
    sigma = np.array(sigma)
    A_k = np.array(A_k)
    b_k = np.array(b_k)
    P = np.array(P)

    return priors, mu, sigma, A_k, b_k, P


def plot_vector_field():
    path_index = 0
    path = lasa.DataSet.BendedLine.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]
    # priors, mu, sigma, A_k, b_k, _ = get_gmm_from_matlab(pos, vel, final_pos)
    priors = np.array([[0.314, 0.193, 0.09, 0.125, 0.146, 0.132]])
    mu = np.array([[-18.24738241,  -2.38762792, -39.88113386, -33.02005433, -16.73786891, -35.41178583],
                   [-2.01251209,   1.94543297,   3.67688337,   9.87653523, 8.73540075,  -1.68948461]])

    sigma = np.array([[[4.72548104e+01,  9.51259945e+00,  5.33200857e+00,
                        1.53709129e+01,  3.45132894e+01,  1.40814127e+01],
                       [-3.76600106e-02, -5.24313331e+00,  6.45669632e-01,
                        3.22841802e+00, -9.24930031e+00, -1.07890577e+00]],
                      [[-3.76600106e-02, -5.24313331e+00,  6.45669632e-01,
                        3.22841802e+00, -9.24930031e+00, -1.07890577e+00],
                       [1.16414861e-01,  3.98247450e+00,  4.98111432e+00,
                          2.14939176e+00,  3.28400643e+00,  8.39037137e-01]]])
    A_k = np.array([[[-4.31045650e-04,  4.18354787e-01,  1.76021431e-01,
                    3.77465310e-01,  7.32518883e-01,  1.21667038e-01],
                    [8.49819063e-05,  3.94922238e+00,  2.76803103e+00,
                    2.97203976e+00,  3.76181358e+00,  4.37518747e+00]],
                    [[8.49819063e-05, -2.27118040e+00, -3.91746552e-01,
                      -6.55744803e-01, -1.08346276e+00, -1.26452215e-01],
                    [-6.89553107e-05, -5.06866106e+00, -5.48327846e-01,
                     -1.69308978e+00, -2.73017118e+00, -2.17499945e+00]]])
    b_k = np.array([[-2.26896828e-10,  1.54856698e-11, -1.07329064e-12,
                    -6.75164833e-13,  1.47433625e-13, -2.30172642e-11],
                    [-2.11879115e-11, -3.58536067e-12, -3.24357875e-13,
                    1.45587833e-12,  4.08763595e-12, -3.29942349e-11]])

    ds_gmm = lpv_ds.GmmVariables(mu, priors, sigma)
    lpvds = lpv_ds.LpvDs(A_k=A_k, b_k=b_k, eps=eps, realmin=realmin)

    fig, ax = plt.subplots(figsize=(10, 8))
    x_lim = [-42.0, 2.0],
    y_lim = [-4.0, 12.0]
    ax.x_lim = x_lim
    ax.y_lim = y_lim
    dim = 2
    # k = 20
    # L = 1000
    # positions_arr = np.zeros((L, dim, k))
    # for i in range(k):
    #     x0 = random.randrange(-40.0, -5.0)
    #     x1 = random.randrange(-2.0, 10.0)
    #     positions_arr[0, :, i] = np.array([x0, x1])
    xs = np.linspace(-40.0, 0.0, 15)
    ys = np.linspace(-5.0, 15.0, 2)
    k = xs.size * ys.size
    L = 1000
    positions_arr = np.zeros((L, dim, k))
    counter = 0
    for x in xs:
        for y in ys:
            positions_arr[0, :, counter] = np.array([x, y])
            counter += 1
    for i in range(k):
        for j in range(L-1):
            x = positions_arr[j, :, i]
            x.shape = (2, 1)
            x_dot = lpvds.evaluate(x, ds_gmm)
            x_dot.shape = (1, 2)
            positions_arr[j+1, :, i] = positions_arr[j, :, i] + \
                0.1 * x_dot / np.linalg.norm(x_dot, 2)
    for i in range(k):
        if i == k-1:
            # color=(random.random(), random.random(), random.random()))
            ax.plot(positions_arr[:, 0, i], positions_arr[:,
                    1, i], markersize=0.5, marker='.', color="gray", label="Computed Trajectories")
        else:
            # color=(random.random(), random.random(), random.random()))
            ax.plot(positions_arr[:, 0, i], positions_arr[:,
                    1, i], markersize=0.5, marker='.', color="gray")
    ax.plot(pos[0, :], pos[1, :], markersize=1.2,
            marker=".", color="black", label='Reference Trajectory')
    ax.plot(pos[0, -1], pos[1, -1], color="black", marker="*",
            markersize=16, label="Attractor Position")
    ax.set_xlim([-47, 16])
    ax.set_ylim([-5, 15])
    ax.legend()
    fig.show()

    input("Press any key to continue...")


def plot_reference_trajectories():
    path_index = 0
    path = lasa.DataSet.LShape.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pos[0, :], pos[1, :], markersize=1.2,
            marker=".", color="black", label='Reference Trajectory')
    ax.plot(pos[0, -1], pos[1, -1], color="black", marker="*",
            markersize=16, label="Attractor Position")
    ax.legend()
    fig.show()

    input("Press any key to continue...")


if (__name__) == "__main__":
    plot_reference_trajectories()

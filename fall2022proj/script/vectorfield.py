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
    priors, mu, sigma, A_k, b_k, _ = get_gmm_from_matlab(pos, vel, final_pos)
    ds_gmm = lpv_ds.GmmVariables(mu, priors, sigma)
    lpvds = lpv_ds.LpvDs(A_k=A_k, b_k=b_k, eps=eps, realmin=realmin)

    fig, ax = plt.subplots(figsize=(10, 8))
    x_lim = [-42.0, 2.0],
    y_lim = [-4.0, 12.0]
    ax.x_lim = x_lim
    ax.y_lim = y_lim
    dim = 2
    k = 20
    L = 1000
    positions_arr = np.zeros((L, dim, k))
    for i in range(k):
        x0 = random.randrange(-40.0, -5.0)
        x1 = random.randrange(-2.0, 10.0)
        positions_arr[0, :, i] = np.array([x0, x1])
    for i in range(k):
        for j in range(L-1):
            x = positions_arr[j, :, i]
            x.shape = (2, 1)
            x_dot = lpvds.evaluate(x, ds_gmm)
            x_dot.shape = (1, 2)
            positions_arr[j+1, :, i] = positions_arr[j, :, i] + \
                0.1 * x_dot / np.linalg.norm(x_dot, 2)
    for i in range(k):
        ax.plot(positions_arr[:, 0, i], positions_arr[:,
                1, i], markersize=0.5, marker='.', color=(random.random(), random.random(), random.random()))
    ax.plot(pos[0, :], pos[1, :], markersize=0.25,
            marker=".", color="yellowgreen")
    fig.show()

    input("Press any key to continue...")


if (__name__) == "__main__":
    plot_vector_field()

import numpy as np
import pyLasaDataset as lasa
import matlab.engine


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


def get_Ak_and_bk_from_gmm(position, velocity, final_pos: np.array, priors, mu, sigma):

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

    priors = matlab.double(priors.astype('float64'))
    mu = matlab.double(mu.astype('float64'))
    sigma = matlab.double(sigma.astype('float64'))

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

    A_k = np.array(A_k)
    b_k = np.array(b_k)
    P = np.array(P)

    return A_k, b_k, P


def load_data(dataset):
    path_index = 0
    if dataset == 'Angle':
        path = lasa.DataSet.Angle.demos[path_index]
    elif dataset == 'BendedLine':
        path = lasa.DataSet.BendedLine.demos[path_index]
    elif dataset == 'CShape':
        path = lasa.DataSet.CShape.demos[path_index]
    elif dataset == 'DoubleBendedLine':
        path = lasa.DataSet.DoubleBendedLine.demos[path_index]
    elif dataset == 'GShape':
        path = lasa.DataSet.GShape.demos[path_index]
    elif dataset == 'LShape':
        path = lasa.DataSet.LShape.demos[path_index]
    elif dataset == 'Sine':
        path = lasa.DataSet.Sine.demos[path_index]
    elif dataset == 'Snake':
        path = lasa.DataSet.Snake.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]
    priors, mus, sigmas, A_k, b_k, _ = get_gmm_from_matlab(pos, vel, final_pos)
    x_lim = [np.min(pos[0, :])-5.0, np.max(pos[0, :])+5.0]
    y_lim = [np.min(pos[1, :])-5.0, np.max(pos[1, :])+5.0]
    return path, pos, vel, priors, mus, sigmas, A_k, b_k, x_lim, y_lim


def load_data_with_predefined_params(dataset):
    path_index = 0
    if dataset == 'Angle':
        path = lasa.DataSet.Angle.demos[path_index]
        pos = path.pos
        vel = path.vel
    elif dataset == 'BendedLine':
        path = lasa.DataSet.BendedLine.demos[path_index]
        pos = path.pos
        vel = path.vel
        priors = np.array([[0.314, 0.193, 0.09, 0.125, 0.146, 0.132]])
        mus = np.array([[-18.24738241,  -2.38762792, -39.88113386, -33.02005433, -16.73786891, -35.41178583],
                       [-2.01251209,   1.94543297,   3.67688337,   9.87653523, 8.73540075,  -1.68948461]])
        sigmas = np.array([[[4.72548104e+01,  9.51259945e+00,  5.33200857e+00,
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
        x_lim = [-45.0, 5.0]
        y_lim = [-5.0, 20.0]
    elif dataset == 'CShape':
        path = lasa.DataSet.CShape.demos[path_index]
        pos = path.pos
        vel = path.vel
    elif dataset == 'DoubleBendedLine':
        path = lasa.DataSet.DoubleBendedLine.demos[path_index]
        pos = path.pos
        vel = path.vel
    elif dataset == 'GShape':
        path = lasa.DataSet.GShape.demos[path_index]
        pos = path.pos
        vel = path.vel
    elif dataset == 'LShape':
        path = lasa.DataSet.LShape.demos[path_index]
        pos = path.pos
        vel = path.vel
        x_lim = [-36.0, 3.0]  # LShape
        y_lim = [-13.0, 48.0]  # LShape
        priors = np.array([[0.164, 0.211, 0.249, 0.165, 0.211]])
        mus = np.array([[-27.97390462, -29.03648456,  -6.1789614, -24.65764882,
                        -31.10150071],
                       [19.59989514,  39.13652236,  -0.7555651,  -2.28375025,
                        4.54848509]])
        sigmas = np.array([[[1.03796892,  1.18131439, 27.08671255, 23.01309662,
                            3.55017427],
                           [1.13976659, -4.22530448,  2.39282625,  0.75962335,
                            7.0116795]],

                          [[1.13976659, -4.22530448,  2.39282625,  0.75962335,
                            7.0116795],
                           [23.72071442, 27.56720204,  2.06683289,  3.2512121,
                              20.04156201]]])
        A_k = np.array([[[-0.14052669,  -0.78031162,  -0.26195104,  -0.5259253,
                        0.04979977],
                        [-0.15149797,  -0.49104621, -21.92820964,  -6.21825116,
                        -1.07633619]],

                        [[-0.15149797,   0.27951289,   0.57422758,   0.69877895,
                          0.41471347],
                        [-1.91330532,  -0.27261989,  -7.22325673,  -6.59743802,
                            -1.53462801]]])
        b_k = np.array([[-5.05414989e-10, -2.33293689e-11,  1.65762129e-09,
                        -4.49657373e-11,  1.32345018e-11],
                        [1.17332605e-11, -5.62835898e-10,  2.36063974e-10,
                        1.67950213e-11, -1.80756117e-11]])
    elif dataset == 'Sine':
        path = lasa.DataSet.Sine.demos[path_index]
        pos = path.pos
        vel = path.vel
    elif dataset == 'Snake':
        path = lasa.DataSet.Snake.demos[path_index]
        pos = path.pos
        vel = path.vel
    return path, pos, vel, priors, mus, sigmas, A_k, b_k, x_lim, y_lim

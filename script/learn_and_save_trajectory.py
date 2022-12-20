import os
import pandas as pd
import numpy as np
from learning_avoidance.load_dataset_and_params import get_gmm_from_matlab
input_file = os.getcwd() + '/data/trajectory2.csv'
output_file = os.getcwd() + '/data/'


def learn_trajector():
    global input_file
    df = pd.read_csv(input_file)
    position = df[['position', 'position.1',
                   'position.2']].to_numpy().transpose()
    velocity = df[['linear_velocity', 'linear_velocity.1',
                   'linear_velocity.2']].to_numpy().transpose()

    priors, mus, sigmas, A_k, b_k, P = get_gmm_from_matlab(
        position, velocity, position[:, -1])

    save_model(priors, mus, sigmas, A_k, b_k, P)
    # np.reshape(B, (B.shape[0], B.shape[0], -1))


def save_model(priors, mus, sigmas, A_k, b_k, P):
    global output_file
    np.savetxt(output_file + 'priors.csv', priors, delimiter=',')
    np.savetxt(output_file + 'mus.csv', mus, delimiter=',')
    np.savetxt(output_file + 'sigmas.csv',
               np.reshape(sigmas, (sigmas.shape[0], -1)), delimiter=',')
    np.savetxt(output_file + 'A_k.csv',
               np.reshape(A_k, (A_k.shape[0], -1)), delimiter=',')
    np.savetxt(output_file + 'b_k.csv', b_k, delimiter=',')
    np.savetxt(output_file + 'P.csv', P, delimiter=',')


if (__name__) == "__main__":
    learn_trajector()

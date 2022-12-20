import numpy as np
import learning_avoidance.lpv_ds as lpv_ds
from learning_avoidance.initial_dynamics import InitialDynamics, InitialDynamicsType
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from numpy import genfromtxt

data_path = './data/'


priors = genfromtxt(data_path+'priors.csv', delimiter=',')
K = priors.size
priors.shape = (1, K)
mus = genfromtxt(data_path+'mus.csv', delimiter=',')
sigmas = genfromtxt(data_path+'sigmas.csv', delimiter=',')
sigmas = np.reshape(sigmas, (sigmas.shape[0], sigmas.shape[0], -1))
A_k = genfromtxt(data_path+'A_k.csv', delimiter=',')
A_k = np.reshape(A_k, (A_k.shape[0], A_k.shape[0], -1))
b_k = genfromtxt(data_path+'b_k.csv', delimiter=',')
P = genfromtxt(data_path+'P.csv', delimiter=',')
gmm_ds = lpv_ds.GmmVariables(mu=mus, priors=priors, sigma=sigmas)
lpvds = lpv_ds.LpvDs(A_k=A_k, b_k=b_k, ds_gmm=gmm_ds)
obstacle_environment = ObstacleContainer()
initial_dynamics = InitialDynamics(
    maximum_velocity=1.0,
    attractor_position=np.array(
        [0.3756374418735504, -0.24511094391345978, 0.3886714279651642]),
    dimension=A_k.shape[0],
)
initial_dynamics.setup(trajectory_dynamics=lpvds,
                       a=100,
                       b=50,
                       obstacle_environment=obstacle_environment,
                       initial_dynamics_type=InitialDynamicsType.WeightedSum,
                       )

x_dot = initial_dynamics.evaluate(np.array([1, 1, 1]))
print(x_dot)

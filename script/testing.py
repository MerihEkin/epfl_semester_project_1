from ssl import AlertDescription
from typing import List

from sympy import false, true
import lpv_ds
import matlab.engine
import numpy as np
from random import random
from random import shuffle

import pyLasaDataset as lasa
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as mplp

from dynamic_obstacle_avoidance.obstacles import Polygon

# from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.containers import ObstacleContainer, obstacle_container

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

import initial_dynamics

from vartools.animator import Animator
from sklearn.mixture import GaussianMixture
from vartools.dynamical_systems import LinearSystem
from random import randrange

eps = 1e-5
realmin = 2e-100
pi = 3.14


class VectorFieldVisualization():
    def __init__(self, x_lim, y_lim, obstacle_environment: ObstacleContainer, reference_path, A_g, b_g, ds_gmm, n_x=20, n_y=20):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.obstacle_environment = obstacle_environment
        self.path = reference_path
        self.nx = n_x
        self.ny = n_y

        lpvds = lpv_ds.LpvDs(A_k=A_g, b_k=b_g, ds_gmm=ds_gmm)

        self.initial_dynamics = initial_dynamics.InitialDynamics(
            maximum_velocity=1.0, attractor_position=reference_path[-1, :])
        self.initial_dynamics.setup(trajectory_dynamics=lpvds,
                                    a=100, b=50, obstacle_environment=obstacle_environment,
                                    initial_dynamics_type=initial_dynamics.InitialDynamicsType.LocallyRotatedFromObstacle,
                                    )

        self.dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=obstacle_environment,
        )

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def plot_vector_field(self):
        self.ax.plot(self.path[:, 0], self.path[:, 1],
                     markersize=0.5, marker=".", color="yellowgreen", zorder=-4)
        xs = np.linspace(self.x_lim[0], self.x_lim[1], self.nx)
        ys = np.linspace(self.y_lim[0], self.y_lim[1], self.ny)
        nr_of_points = xs.size * ys.size
        x = np.zeros(nr_of_points)
        y = np.zeros(nr_of_points)
        u = np.zeros(nr_of_points)
        v = np.zeros(nr_of_points)
        counter = 0
        for i in xs:
            for j in ys:
                x[counter] = i
                y[counter] = j
                pos = np.array([i, j])
                dir = self.dynamic_avoider.evaluate(position=pos)
                u[counter] = dir[0]
                v[counter] = dir[1]
                counter += 1
        self.ax.quiver(x, y, u, v, color='black', zorder=-1)
        # Draw obstacles
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
            alpha_obstacle=1.0,
            linealpha=1.0,
        )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.show()
        input("Enter key to continue...")


def visualize_vector_field():
    path_index = 0
    path = lasa.DataSet.BendedLine.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]

    x_lim = [-45.0, 5.0]
    y_lim = [-5.0, 20.0]

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

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[5.0, 15.0],
            center_position=np.array([-16.0, 10.0]),
            margin_absolut=0.5,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Ellipse(
            axes_length=[5.0, 5.0],
            center_position=np.array([-38.0, 7.5]),
            margin_absolut=0.5,
            tail_effect=False,
        )
    )
    vfv = VectorFieldVisualization(
        x_lim=x_lim,
        y_lim=y_lim,
        obstacle_environment=obstacle_environment,
        reference_path=pos.transpose(),
        n_x=40, n_y=40,
        A_g=A_k, b_g=b_k,
        ds_gmm=ds_gmm,
    )
    vfv.plot_vector_field()


if (__name__) == "__main__":
    visualize_vector_field()

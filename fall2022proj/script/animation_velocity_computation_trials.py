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

from dynamic_obstacle_avoidance.containers import ObstacleContainer, obstacle_container

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.animator import Animator
from sklearn.mixture import GaussianMixture
from vartools.dynamical_systems import LinearSystem
from random import randrange

eps = 1e-5
realmin = 2e-100
pi = 3.14


class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        start_position,
        ds_gmm: lpv_ds.GmmVariables,
        A_g,
        b_g,
        obstacle_environment: ObstacleContainer,
        obstacle_targets: List,
        reference_path,
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.start_position = start_position
        self.ds_gmm = ds_gmm
        self.A_g = A_g
        self.b_g = b_g
        self.obstacle_environment = obstacle_environment
        self.obstacle_targets = obstacle_targets
        self.reference_path = reference_path
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.agent_dynamics = lpv_ds.LpvDs(
            A_k=self.A_g, b_k=self.b_g,  target=np.array([0.0, 0.0]), obstacle_environment=self.obstacle_environment,
            eps=eps, realmin=realmin, algorithm=lpv_ds.Algorithm.Weighted1)

        self.agent_dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.agent_dynamics,
            obstacle_environment=self.obstacle_environment,
        )

        assert (len(self.obstacle_targets) <= len(
            self.obstacle_environment._obstacle_list))

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self.agent_positions = np.zeros((self.dim, self.it_max))
        self.agent_positions[:, 0] = self.start_position

        self.K = ds_gmm.mu.shape[1]
        self.attractor_dynamics = []
        self.dynamic_avoiders = []
        self.attractor_colors = []
        self.attractor_positions = np.zeros((self.dim, self.it_max, self.K))
        for i in range(self.K):
            self.attractor_dynamics.append(
                LinearSystem(
                    attractor_position=np.array(
                        [self.ds_gmm.mu[0, i], self.ds_gmm.mu[1, i]]),
                    maximum_velocity=1,
                    distance_decrease=0.3,
                )
            )
            self.dynamic_avoiders.append(
                ModulationAvoider(
                    initial_dynamics=self.attractor_dynamics[i], obstacle_environment=self.obstacle_environment,
                )
            )
            self.attractor_positions[:, 0, i] = self.ds_gmm.mu[:, i]
            self.attractor_colors.append((random(), random(), random()))
            self.max_vel = 2.0

    def update_step(self, ii):

        # Update attractors
        for i in range(self.K):
            velocity = self.dynamic_avoiders[i].evaluate(
                self.attractor_positions[:, ii-1, i])
            self.attractor_positions[:, ii, i] = self.attractor_positions[:,
                                                                          ii-1, i] + velocity * self.dt_simulation
            self.ds_gmm.mu[:, i] = self.attractor_positions[:, ii, i]

        # Update agent
        if self.agent_dynamics.algorithm == lpv_ds.Algorithm.Plain:
            x_dot = self.agent_dynamic_avoider.evaluate_with_gmm(
                self.agent_positions[:, ii-1], self.ds_gmm)
        elif self.agent_dynamics.algorithm == lpv_ds.Algorithm.Weighted1 or self.agent_dynamics.algorithm == lpv_ds.Algorithm.Weighted2:
            x = self.agent_positions[:, ii-1]
            x.shape = (2, 1)
            x_dot = self.agent_dynamics.evaluate(x, self.ds_gmm)
        if (np.linalg.norm(x_dot) > self.max_vel):
            x_dot = self.max_vel * x_dot / np.linalg.norm(x_dot)
        x_dot.shape = (2,)
        self.agent_positions[:, ii] = (
            2.0 * self.dt_simulation * x_dot + self.agent_positions[:, ii-1])

        # Update obstacles
        for i in range(len(self.obstacle_targets)):
            target_pos = self.obstacle_targets[i]
            current_pos = self.obstacle_environment._obstacle_list[i].center_position
            dst = target_pos - current_pos
            if (np.linalg.norm(dst, 2) < 0.5 and not np.allclose(self.obstacle_environment._obstacle_list[i].linear_velocity, np.array([0.0, 0.0]))):
                self.obstacle_environment._obstacle_list[i].linear_velocity = np.array([
                    0.0, 0.0])
            elif (np.linalg.norm(dst, 2) < 0.5):
                pass
            else:
                self.obstacle_environment._obstacle_list[i].linear_velocity = dst / np.linalg.norm(
                    dst, 2)

        self.obstacle_environment.do_velocity_step(
            delta_time=self.dt_simulation)

        # Draw ref path
        self.ax.clear()
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.plot(self.reference_path[:, 0], self.reference_path[:, 1],
                     markersize=0.25, marker=".", color="yellowgreen")

        # Draw obstacles
        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
        )

        # Draw attractor regions
        self.plot_attractors(ii)

        # Draw the agent
        self.plot_agent(ii)

        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def plot_attractors(self, ii):
        for i in range(self.K):
            covariances = self.ds_gmm.sigma[:, :, i]
            center = self.ds_gmm.mu[:, i]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mplp.Ellipse(
                center, v[0], v[1], 180 + angle)
            ell.set_alpha(0.25)
            ell.set_facecolor(self.attractor_colors[i])
            self.ax.add_artist(ell)
            self.ax.plot(self.ds_gmm.mu[0, i],
                         self.ds_gmm.mu[1, i], "k*", markersize=8)

    def plot_agent(self, ii):
        self.ax.plot(
            self.agent_positions[0, :ii], self.agent_positions[1, :ii], color="#135e08",)
        self.ax.plot(
            self.agent_positions[0, ii], self.agent_positions[1, ii], "o", color="#135e08", markersize=12,)

    def has_converged(self, ii) -> bool:
        return False


def run_simulation():
    path_index = 0
    path = lasa.DataSet.BendedLine.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]

    x_lim = [-55.0, 15.0]
    y_lim = [-10.0, 25.0]

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

    # obstacle_environment.append(
    #     Ellipse(
    #         axes_length=[5.0, 5.0],
    #         center_position=np.array([-38.0, 7.5]),
    #         margin_absolut=0.5,
    #         tail_effect=False,
    #     )
    # )

    obstacle_targets = []

    my_animation = DynamicalSystemAnimation(
        it_max=10000,
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        # start_position=pos[:, 0],
        start_position=np.array([-38.0, -1.5]),
        ds_gmm=ds_gmm,
        A_g=A_k,
        b_g=b_k,
        obstacle_environment=obstacle_environment,
        obstacle_targets=obstacle_targets,
        reference_path=pos.transpose(),
        x_lim=x_lim,
        y_lim=y_lim,
    )

    my_animation.run()


if (__name__) == "__main__":
    run_simulation()

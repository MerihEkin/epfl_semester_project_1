from typing import List

from sympy import false, true
import lpv_ds
import numpy as np
from random import random
from random import shuffle

import pyLasaDataset as lasa
import matplotlib.pyplot as plt
import matplotlib.patches as mplp
import load_dataset_and_params as dataloader

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
            A_k=A_g, b_k=b_g, ds_gmm=ds_gmm
        )

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
        x_dot = self.agent_dynamic_avoider.evaluate(
            self.agent_positions[:, ii-1])

        # x_dot = (x_dot / np.linalg.norm(x_dot, 2))
        if (np.linalg.norm(x_dot) > self.max_vel):
            x_dot = self.max_vel * x_dot / np.linalg.norm(x_dot)
        x_dot.shape = (1, 2)
        self.agent_positions[:, ii] = (
            self.dt_simulation * x_dot + self.agent_positions[:, ii-1])

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
                     linewidth='3.0', color="dimgray", label="Reference Trajectory")

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
        self.ax.plot(self.reference_path[-1, 0], self.reference_path[-1, 1],
                     marker="X", markersize=12.0, color="black", label="Attractor")
        self.ax.legend()
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
            self.agent_positions[0, :ii], self.agent_positions[1, :ii], linewidth='3.0', color="#135e08", label="Computed Trajectory")
        self.ax.plot(
            self.agent_positions[0, ii], self.agent_positions[1, ii], "o", color="#135e08", markersize=12,)

    def has_converged(self, ii) -> bool:
        return False


def run_simulation():
    _, pos, _, priors, mus, sigmas, A_k, b_k, x_lim, y_lim = dataloader.load_data_with_predefined_params(
        'BendedLine')

    ds_gmm = lpv_ds.GmmVariables(mus, priors, sigmas)

    obstacle_environment = ObstacleContainer()

    obstacle_environment.append(
        Cuboid(
            axes_length=[5.0, 15.0],
            center_position=np.array([-12.0, 28.0]),
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

    obstacle_targets = []
    obstacle_targets.append(np.array([-16.0, 10.0]))
    # idx = list(range(mu.shape[1]))
    # shuffle(idx)
    # for i in idx:
    #     obstacle_targets.append(np.array([mu[0, i], mu[1, i]]))
    #     if (len(obstacle_targets) == len(obstacle_environment._obstacle_list) or len(obstacle_targets) == mu.shape[1]):
    #         break

    my_animation = DynamicalSystemAnimation(
        it_max=10000,
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        start_position=np.array([-37.4, -2.2]),
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

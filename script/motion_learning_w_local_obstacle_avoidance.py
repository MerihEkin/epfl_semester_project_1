
import lpv_ds
from math import exp
from tkinter import W
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import load_dataset_and_params as dataloader

from vartools.dynamical_systems._base import DynamicalSystem
from vartools.states import ObjectPose
import dynamic_obstacle_avoidance.obstacle_linear_dynamics as ObstacleLinearDynamics
from dynamic_obstacle_avoidance.containers import ObstacleContainer
import vartools.directional_space.directional_space as DirectionalSpace
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.avoidance.modulation import obs_avoidance_interpolation_moving


class InitialDynamicsType(Enum):
    LocalTangentApprox = 1
    LocalAttractorFollowing = 2
    LocalAvoidanceWLinearization = 3
    LocallyRotatedFromObstacle = 4


class MotionLearningWAvoidance():
    def __init__(self, trajectory_dynamics: DynamicalSystem, a, b,
                 obstacle_environment: ObstacleContainer,
                 apply_obstacle_avoidance_after_weighting=True,
                 initial_dynamics_type: InitialDynamicsType = InitialDynamicsType.LocalAvoidanceWLinearization):

        self.trajectory_dynamics = trajectory_dynamics
        self.obstacle_environment = obstacle_environment
        self.a = a
        self.b = b
        self.apply_obstacle_avoidance_after_weighting = apply_obstacle_avoidance_after_weighting
        self.initial_dynamics_type = initial_dynamics_type

    def compute_alpha(self, position):
        N_obs = len(self.obstacle_environment._obstacle_list)
        if not N_obs:
            return 0, np.zeros((position.size)), None

        gammas = np.zeros((N_obs))

        for n in range(N_obs):
            gammas[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)
            if gammas[n] < 1:
                return -1, None, None

        min_gamma = np.min(gammas)
        index = np.argmin(gammas)

        alpha = 1 / (1 + exp(-self.a * (1/(min_gamma+1e-16)) + self.b))

        obs = self.obstacle_environment._obstacle_list[index]

        obs_center = np.array(obs.center_position, copy=True)

        return alpha, obs_center, obs

    def evaluate(self, position, attractor):
        vel_lpvds = self.trajectory_dynamics.evaluate(position)

        alpha, obs_center, obs = self.compute_alpha(position)

        if alpha == -1:
            return np.array([0.0, 0.0])

        if not obs:
            return vel_lpvds

        if self.initial_dynamics_type == InitialDynamicsType.LocalAttractorFollowing:
            vel_initial = (attractor - position)
            vel_initial /= np.linalg.norm(vel_initial, 2)
            vel_avoidance = np.array(vel_initial, copy=True)
            vel_avoidance = obs_avoidance_interpolation_moving(
                position, vel_initial, self.obstacle_environment
            )
            return ((1-alpha)*vel_lpvds + alpha * vel_avoidance)
        elif self.initial_dynamics_type == InitialDynamicsType.LocalTangentApprox:
            # alpha, obs_center, obs = self.compute_alpha(position)
            # ps = obs.pose.transform_position_from_reference_to_local(position)
            # ax = np.copy(obs.semiaxes)
            # theta = obs.orientation_in_degree * np.pi / 180
            # a_ = ax[0]*cos(theta) + ax[1]*sin(theta)
            # b_ = ax[0]*sin(theta) + ax[1]*cos(theta)
            pass
        elif self.initial_dynamics_type == InitialDynamicsType.LocalAvoidanceWLinearization:

            vel_initial = self.trajectory_dynamics.evaluate(obs_center)

            null_direction = attractor - position
            null_direction /= np.linalg.norm(null_direction, 2)

            if self.apply_obstacle_avoidance_after_weighting:
                directions = np.vstack((vel_lpvds, vel_initial))
                directions = directions.T
                weights = np.array([(1-alpha), alpha])
                vel_reference = DirectionalSpace.get_directional_weighted_sum(
                    null_direction=null_direction, directions=directions, weights=weights, normalize=True)
                vel = obs_avoidance_interpolation_moving(
                    position, vel_reference, self.obstacle_environment)
                return vel
            else:
                vel_avoidance = obs_avoidance_interpolation_moving(
                    position, vel_initial, self.obstacle_environment)
                directions = np.vstack((vel_lpvds, vel_avoidance))
                directions = directions.T
                weights = np.array([(1-alpha), alpha])
                vel = DirectionalSpace.get_directional_weighted_sum(
                    null_direction=null_direction, directions=directions, weights=weights, normalize=False)
                return vel

        elif self.initial_dynamics_type == InitialDynamicsType.LocallyRotatedFromObstacle:

            vel_initial = self.trajectory_dynamics.evaluate(obs_center)
            avoidance_dynamics = ObstacleLinearDynamics.LocallyRotatedFromObtacle(
                obstacle=obs, attractor_position=attractor, reference_velocity=vel_initial)
            vel_initial = avoidance_dynamics.evaluate(position=position)

            null_direction = attractor - position
            null_direction /= np.linalg.norm(null_direction, 2)

            if self.apply_obstacle_avoidance_after_weighting:
                directions = np.vstack((vel_lpvds, vel_initial))
                directions = directions.T
                weights = np.array([(1-alpha), alpha])
                vel_reference = DirectionalSpace.get_directional_weighted_sum(
                    null_direction=null_direction, directions=directions, weights=weights, normalize=True)
                vel = obs_avoidance_interpolation_moving(
                    position, vel_reference, self.obstacle_environment)
                return vel
            else:
                vel_avoidance = obs_avoidance_interpolation_moving(
                    position, vel_initial, self.obstacle_environment)
                directions = np.vstack((vel_lpvds, vel_avoidance))
                directions = directions.T
                weights = np.array([(1-alpha), alpha])
                vel = DirectionalSpace.get_directional_weighted_sum(
                    null_direction=null_direction, directions=directions, weights=weights, normalize=False)
                return vel


class VectorFieldVisualization():
    def __init__(self, x_lim, y_lim, obstacle_environment: ObstacleContainer, reference_path, A_g, b_g, ds_gmm, n_x=20, n_y=20):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.obstacle_environment = obstacle_environment
        self.path = reference_path
        self.nx = n_x
        self.ny = n_y

        lpvds = lpv_ds.LpvDs(A_k=A_g, b_k=b_g, ds_gmm=ds_gmm)

        self.trajectory_dynamics_with_local_avoidance = MotionLearningWAvoidance(trajectory_dynamics=lpvds,
                                                                                 a=100, b=50,
                                                                                 obstacle_environment=self.obstacle_environment,
                                                                                 apply_obstacle_avoidance_after_weighting=True,
                                                                                 initial_dynamics_type=InitialDynamicsType.LocallyRotatedFromObstacle,
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
                dir = self.trajectory_dynamics_with_local_avoidance.evaluate(
                    position=pos, attractor=self.path[-1, :])
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
    _, pos, _, priors, mus, sigmas, A_k, b_k, x_lim, y_lim = dataloader.load_data_with_predefined_params(
        'BendedLine')

    ds_gmm = lpv_ds.GmmVariables(mus, priors, sigmas)

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

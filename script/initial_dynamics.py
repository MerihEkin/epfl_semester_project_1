
import lpv_ds
from math import exp
from tkinter import W
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
# import load_dataset_and_params as dataloader

from vartools.dynamical_systems._base import DynamicalSystem
from vartools.states import ObjectPose
import dynamic_obstacle_avoidance.obstacle_linear_dynamics as ObstacleLinearDynamics
from dynamic_obstacle_avoidance.containers import ObstacleContainer
import vartools.directional_space.directional_space as DirectionalSpace
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles


class InitialDynamicsType(Enum):
    LocalTangentApprox = 1
    LocalAttractorFollowing = 2
    LocalAvoidanceWLinearization = 3
    LocallyRotatedFromObstacle = 4
    WeightedSum = 5


class InitialDynamics(DynamicalSystem):
    def __init__(self, pose: ObjectPose = None, maximum_velocity: float = None, dimension: int = None, attractor_position: np.ndarray = None):
        super().__init__(pose, maximum_velocity, dimension, attractor_position)

    def setup(self, trajectory_dynamics: DynamicalSystem, a, b,
              obstacle_environment: ObstacleContainer, distance_decrease: float = 1.0,
              initial_dynamics_type: InitialDynamicsType = InitialDynamicsType.WeightedSum) -> None:
        self.trajectory_dynamics = trajectory_dynamics
        self.obstacle_environment = obstacle_environment
        self.initial_dynamics_type = initial_dynamics_type
        self.distance_decrease = distance_decrease
        self.a = a
        self.b = b

    def evaluate(self, position) -> np.array:

        vel_lpvds = self.trajectory_dynamics.evaluate(position)

        null_direction = self._attractor_position - position
        null_direction /= np.linalg.norm(null_direction, 2)

        if self.initial_dynamics_type == InitialDynamicsType.WeightedSum:
            weights, N_obs = self.compute_weights(position)

            if weights is None:
                return vel_lpvds

            if weights is False:
                return np.zeros((self.dimension))

            directions = np.zeros((N_obs+1, self.dimension))
            for n in range(N_obs):
                obs_center = self.obstacle_environment._obstacle_list[n].center_position
                obs_center = obs_center.copy()
                directions[n, :] = self.trajectory_dynamics.evaluate(
                    obs_center)
                if True:
                    vel_reference = np.array(directions[n, :], copy=True)
                    LRFO = ObstacleLinearDynamics.LocallyRotatedFromObtacle(
                        obstacle=self.obstacle_environment._obstacle_list[n],
                        attractor_position=self.attractor_position,
                        reference_velocity=vel_reference,
                    )
                    directions[n, :] = LRFO.evaluate(position=position)

            directions[-1, :] = vel_lpvds
            directions = directions.T

            velocity = DirectionalSpace.get_directional_weighted_sum(
                null_direction=null_direction, directions=directions, weights=weights, normalize=True)

            velocity = self.limit_velocity(velocity=velocity)

            return velocity

        alpha, obs_center, obs = self.compute_alpha(position)

        if not alpha:
            return np.zeros(self.dimension)

        if obs is None:
            return vel_lpvds

        weights = np.array([(1-alpha), alpha])

        if self.initial_dynamics_type == InitialDynamicsType.LocalTangentApprox:
            pass
        elif self.initial_dynamics_type == InitialDynamicsType.LocalAttractorFollowing:
            directions = np.vstack((vel_lpvds, null_direction))
            directions = directions.T
        elif self.initial_dynamics_type == InitialDynamicsType.LocalAvoidanceWLinearization:
            vel_linearization = self.trajectory_dynamics.evaluate(obs_center)
            directions = np.vstack((vel_lpvds, vel_linearization))
            directions = directions.T
        elif self.initial_dynamics_type == InitialDynamicsType.LocallyRotatedFromObstacle:
            vel_initial = self.trajectory_dynamics.evaluate(obs_center)
            LRFO = ObstacleLinearDynamics.LocallyRotatedFromObtacle(
                obstacle=obs, attractor_position=self.attractor_position, reference_velocity=vel_initial)
            vel_initial = LRFO.evaluate(position=position)
            directions = np.vstack((vel_lpvds, vel_initial))
            directions = directions.T

        velocity = DirectionalSpace.get_directional_weighted_sum(
            null_direction=null_direction, directions=directions, weights=weights, normalize=True)

        velocity = self.limit_velocity(velocity=velocity)
        velocity = self.limit_velocity_around_attractor(
            velocity=velocity, position=position)

        return velocity

    def compute_alpha(self, position):
        N_obs = len(self.obstacle_environment._obstacle_list)

        if N_obs == 0:
            return 0, None, None

        gammas = np.zeros((N_obs))

        for n in range(N_obs):
            gammas[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)
            if gammas[n] <= 1:   # comment
                return False, None, None

        min_gamma = np.min(gammas)
        index = np.argmin(gammas)

        alpha = 1 / (1 + exp(-self.a * (1/(min_gamma+1e-16)) + self.b))

        obs = self.obstacle_environment._obstacle_list[index]

        obs_center = np.array(obs.center_position, copy=True)

        return alpha, obs_center, obs

    def compute_weights(self, position):
        N_obs = len(self.obstacle_environment._obstacle_list)

        if N_obs == 0:
            return None, N_obs

        gammas = np.zeros((N_obs))
        weights = np.zeros((N_obs+1))

        for n in range(N_obs):
            gammas[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)
            if gammas[n] < 1:   # comment
                return False, None
            elif abs(gammas[n] - 1) < 1e-3:
                weights = np.zeros((N_obs))
                weights[n] = 1
                break
            weights[n] = 1/(gammas[n] - 1)

        sum_weights = np.sum(weights)
        if sum_weights > 1:
            weights /= sum_weights
        else:
            weights[-1] = 1 - sum_weights

        return weights, N_obs

    def limit_velocity_around_attractor(self, velocity, position):
        dist_attr = np.linalg.norm(position - self.attractor_position)

        if not dist_attr:
            return np.zeros(velocity.shape)

        mag_vel = np.linalg.norm(velocity)
        if not mag_vel:
            return velocity

        if dist_attr > self.distance_decrease:
            desired_velocity = self.maximum_velocity
        else:
            desired_velocity = self.maximum_velocity * (
                dist_attr / self.distance_decrease
            )
        return velocity / mag_vel * desired_velocity


class VectorFieldVisualization():
    def __init__(self, x_lim, y_lim, obstacle_environment: ObstacleContainer, reference_path, A_g, b_g, ds_gmm, n_x=20, n_y=20):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.obstacle_environment = obstacle_environment
        self.path = reference_path
        self.nx = n_x
        self.ny = n_y

        lpvds = lpv_ds.LpvDs(A_k=A_g, b_k=b_g, ds_gmm=ds_gmm)

        self.initial_dynamics = InitialDynamics(
            maximum_velocity=1.0,
            attractor_position=reference_path[-1, :],
            dimension=2,
        )
        self.initial_dynamics.setup(trajectory_dynamics=lpvds,
                                    a=100,
                                    b=50,
                                    obstacle_environment=obstacle_environment,
                                    initial_dynamics_type=InitialDynamicsType.LocallyRotatedFromObstacle,
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
                # dir = self.initial_dynamics.evaluate(position=pos)
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


from math import exp
from tkinter import W
import numpy as np
from enum import Enum

from vartools.dynamical_systems._base import DynamicalSystem
import dynamic_obstacle_avoidance.obstacle_linear_dynamics as old
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.avoidance.modulation import obs_avoidance_interpolation_moving


class ObstacleAvoidanceType(Enum):
    ModulationAvoider = 1
    LocallyRotatedFromObstacle = 2


class InitialDynamicsType(Enum):
    LocalTangentApprox = 1
    LocalAttractorFollowing = 2
    LocalAvoidanceWLinearization = 3


class MotionLearningWAvoidance():
    def __init__(self, trajectory_dynamics: DynamicalSystem, a, b,
                 obstacle_environment: ObstacleContainer,
                 apply_obstacle_avoidance_after_weighting=True,
                 avoidance_dynamics_type:  ObstacleAvoidanceType = ObstacleAvoidanceType.ModulationAvoider,
                 initial_dynamics_type: InitialDynamicsType = InitialDynamicsType.LocalAvoidanceWLinearization):

        self.trajectory_dynamics = trajectory_dynamics
        self.obstacle_environment = obstacle_environment
        self.a = a
        self.b = b
        self.apply_obstacle_avoidance_after_weighting = apply_obstacle_avoidance_after_weighting
        self.avoidance_dynamics_type = avoidance_dynamics_type
        self.initial_dynamics_type = initial_dynamics_type

    def compute_alpha(self, position):
        # return alpha, obs_center, obs
        N_obs = len(self.obstacle_environment._obstacle_list)
        if not N_obs:
            return 0, np.zeros((position.size)), None

        gammas = np.zeros((N_obs))

        for n in range(N_obs):
            gammas[n] = self.obstacle_environment._obstacle_list[n].get_gamma(
                position, in_obstacle_frame=False, in_global_frame=True)

        min_gamma = np.min(gammas)
        index = np.argmin(gammas)

        alpha = 1 / (1 + exp(-self.a * (1/(min_gamma+1e-16)) + self.b))

        obs = self.obstacle_environment._obstacle_list[index]

        obs_center = np.array(obs.center_position, copy=True)

        return alpha, obs_center, obs

    def evaluate(self, position, attractor):
        vel_lpvds = self.trajectory_dynamics.evaluate(position)

        if self.initial_dynamics_type == InitialDynamicsType.LocalAttractorFollowing:
            alpha, _, _ = self.compute_alpha(position)
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
            alpha, obs_center, obs = self.compute_alpha(position)
            vel_initial = self.trajectory_dynamics.evaluate(obs_center)
            vel_avoidance = np.array(vel_initial, copy=True)

            if self.avoidance_dynamics_type == ObstacleAvoidanceType.ModulationAvoider:
                if self.apply_obstacle_avoidance_after_weighting:
                    vel_initial = (1-alpha)*vel_lpvds + alpha * vel_initial
                    vel = obs_avoidance_interpolation_moving(
                        position, vel_initial, self.obstacle_environment
                    )
                    return vel
                else:
                    vel_avoidance = obs_avoidance_interpolation_moving(
                        position, vel_initial, self.obstacle_environment
                    )
                    return ((1-alpha)*vel_lpvds + alpha * vel_avoidance)

            elif self.avoidance_dynamics_type == ObstacleAvoidanceType.LocallyRotatedFromObstacle:
                if self.apply_obstacle_avoidance_after_weighting:
                    vel_initial = (1-alpha)*vel_lpvds + alpha * vel_initial
                    avoidance_dynamics = old.LocallyRotatedFromObtacle(
                        obstacle=obs, attractor_position=attractor, reference_velocity=vel_initial)
                    vel = avoidance_dynamics.evaluate(position=position)
                    return vel
                else:
                    avoidance_dynamics = old.LocallyRotatedFromObtacle(
                        obstacle=obs, attractor_position=attractor, reference_velocity=vel_initial)
                    vel_avoidance = avoidance_dynamics.evaluate(
                        position=position)
                    return ((1-alpha)*vel_lpvds + alpha * vel_avoidance)

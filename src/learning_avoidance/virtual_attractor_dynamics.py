import warnings

import numpy as np
import numpy.linalg as LA

from vartools.dynamical_systems._base import DynamicalSystem

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from .lpv_ds import LpvDs
from .lpv_ds import GmmVariables


class VirtualAttractorDynamics(DynamicalSystem):
    def __init__(
        self, attractor_position: np.ndarray, lpvds: LpvDs, obstacles: ObstacleContainer
    ):
        self.attractor_position = attractor_position

        # The real one is to store the absolute values
        # -> the virtual one to evaluate the system (with approaching of the identity matrix)
        self._real_lpvds = lpvds
        self._virtual_system = lpvds

        self._obstacles = obstacles

        self.obstacle_attractors = np.array(
            (self.obstacles.dimension, len(self._obstacles))
        )

        # Attractor and position distances
        self.attractor_distances = np.array(len(self._obstacles))
        self.position_distances = np.array(len(self._obstacles))
        self.weights = np.array(len(self._obstacles))

    def update_main_virtual_attractor(self, position):
        for kk in range(len(self.obstacles)):
            self._update_obstacle_virtual_attractors(kk, position)

        # Compute redistribution weights
        ind_border = self.position_distances >= 1
        if num_border := np.sum(ind_border):
            if num_border > 1:
                warnings.warn("Many and intersecting obstacles.")
                normalized_weights = ind_border / num_border
            else:
                normalized_weights = num_border

            weight_sum = 1
        else:
            # TODO: create distance cut-off
            normalized_weights = 1 / (1 - self.weights) - 1

            if not (weight_sum := np.sum(normalized_weights)):
                self.virtual_attractor = self.position

            if weight_sum > 1:
                normalized_weights = self.weights / weight_sum
                weight_sum = 1

        # Position of weights
        breakpoint()
        self.virtual_attractor = np.copy(self.attractor_position) * (1 - weight_sum)
        self.virtual_attractor = np.sum(
            self.obstacle_attractors
            * np.tile(normalized_weights, (1, self.dimension)).T,
            axis=1,
        )

        # In order to reach the attractor, the position (by design) has to be on or outside of the attractor region
        # TODO: we need to check if this is really needed...
        mean_attractor_distance = np.sum(self.attractor_distances * normalized_weights)
        dist_attractor = LA.norm(self.virtual_attractor - position)
        if mean_attractor_distance < dist_attractor:
            # If we are getting close, we ensure that the attractor is moving towards the real-attractor
            closeness_weight = dist_attractor / mean_attractor_distance

            self.virtual_attractor = (
                closeness_weight * self.virtual_attractor
                + (1 - closeness_weight) * self.attractor_position
            )

        # Update the virtual system
        self._virtual_system.A_k = self._real_lpvds.A_k * (1 - weight_sum)
        for kk in range(self.n_elements):
            self._virtual_system.b_k = self.get_A_k(kk) @ self.virtual_attractor

    def evaluate(self, position):
        self.update_main_virtual_attractor(position)
        return self._virtual_system.evaluate(position)

    def _update_obstacle_virtual_attractors(self, index, position):
        obstacle = self.obstacles[index]

        # Distance to attractor
        attractor_dir = self.attractor_position - obstacle.center_position

        # Local radius is also the maximum-influence circle
        attractor_distance = obstacle.get_local_radius(
            attractor_dir, in_global_frame=True
        )

        attractor_gamma = obstacle.get_gamma(
            self.attractor_position, in_global_frame=True
        )

        if not (norm_dir := LA.norm(attractor_dir)):
            raise NotImplementedError("Implement for obstacle at center.")
        attractor_dir = attractor_dir / norm_dir

        local_velocity = self._lpvds.evaluate(obstacle.center_position)
        if not (norm_vel := LA.norm(local_velocity)):
            raise NotImplementedError("Implement for zero velocity.")

        center_ds_direction = local_velocity / norm_vel

        # position_gamma = obstacle.get_gamma(position, in_global_frame=True)
        position_distance = LA.norm(
            position - obstacle.center_position
        ) - obstacle.get_local_radius(position, in_global_frame=True)

        if position_distance > attractor_distance:
            return self.attractor_position

        if position_distance <= -1:
            return obstacle.center_position + center_ds_direction * attractor_distance

        # weight: 1 -> (real) attractor
        virtual_attractor_weight = position_distance - attractor_distance
        virtual_attractor_direction = (
            virtual_attractor_weight * attractor_dir
            + (1 - virtual_attractor_weight) * center_ds_direction
        )
        virtual_attractor_direction = virtual_attractor_direction / LA.norm(
            virtual_attractor_direction
        )
        return (
            obstacle.center_position + virtual_attractor_direction * attractor_distance
        )

    def get_mu(self, index):
        return self._lpvds.ds_gmm.mu[index, :]

    def get_prior(self, index):
        return self._lpvds.ds_gmm.priors[index]

    def get_sigma(self, index):
        return self._lpvds.ds_gmm.sigma[index, :, :]

    def get_A_k(self, index):
        return self._virtual_system.A_k[:, :, index]

    def get_b_k(self, index):
        return self._lpvds.b_k[:, index]

    @property
    def n_elements(self) -> int:
        return self._virtual_system.ds_gmm.priors.shape[1]


def _test_lpvds():

    mu = []
    priors = []
    sigma = []

    A_k = []
    b_k = []

    attractor_position = np.array([0, 0])

    mu.append([2, 0])
    priors.append(1)
    sigma.append([[1, 0], [0, 1]])

    A_k.append([[-1, 0.5], [0.5, -1]])
    b_k.append(np.array(A_k[-1]) @ attractor_position)

    gamma_variables = GmmVariables(mu, priors, sigma)

    lpv_ds = LpvDs(A_k, b_k, gamma_variables)

    dynamics = VirtualAttractorDynamics(lpv_ds=lpv_ds)

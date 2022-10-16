import time
import os
import datetime
from math import pi

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

from dynamic_obstacle_avoidance.obstacles import Polygon

# from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator


class DSAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        initial_positions,
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.dynamic_avoiders = []
        self.position_list = []

        for i in range(0, len(self.initial_dynamics)):
            self.dynamic_avoiders.append(
                ModulationAvoider(
                    initial_dynamics=self.initial_dynamics[i], obstacle_environment=self.obstacle_environment)
            )
            self.position_list.append(
                np.zeros((self.dim, self.it_max))
            )
            self.position_list[i][:, 0] = np.array(
                [initial_positions[i][0], initial_positions[i][1]])

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        # Here come the main calculation part
        for i in range(0, len(self.initial_dynamics)):
            velocity = self.dynamic_avoiders[i].evaluate(
                self.position_list[i][:, ii - 1])

            self.position_list[i][:, ii] = (
                velocity * self.dt_simulation +
                self.position_list[i][:, ii - 1]
            )

        # Update obstacles
        self.obstacle_environment.do_velocity_step(
            delta_time=self.dt_simulation)

        self.ax.clear()

        for i in range(0, len(self.initial_dynamics)):
            # Drawing and adjusting of the axis
            self.ax.plot(
                self.position_list[i][0, :ii], self.position_list[i][1, :ii], ":", color="#135e08"
            )
            self.ax.plot(
                self.position_list[i][0, ii],
                self.position_list[i][1, ii],
                "o",
                color="#135e08",
                markersize=12,
            )

            self.ax.plot(
                self.initial_dynamics[i].attractor_position[0],
                self.initial_dynamics[i].attractor_position[1],
                "k*",
                markersize=8,
            )

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
        )

        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        for ps in self.position_list:
            if not np.allclose(ps[:, ii], ps[:, ii-1]):
                return False
        return True


def run_multi_agent_obstacle_avoidance():
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.5, 0.5],
            # center_position=np.array([-3.0, 0.2]),
            center_position=np.array([-2.0, -1.5]),
            margin_absolut=0.5,
            orientation=0,
            linear_velocity=np.array([0.35, 0.35]),
            tail_effect=False,
        )
    )

    initial_dynamics = []

    initial_positions = [(0.0, 0.0), (0.0, 1.0), (-2.0, -0.5), (-1.0, 0.0)]

    for ip in initial_positions:
        initial_dynamics.append(
            LinearSystem(
                attractor_position=np.array([ip[0], ip[1]]),
                maximum_velocity=1,
                distance_decrease=0.3,
            )
        )

    my_animation = DSAnimation(
        it_max=200,
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        initial_dynamics, obstacle_environment, initial_positions)
    my_animation.run(save_animation=False)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()
    run_multi_agent_obstacle_avoidance()

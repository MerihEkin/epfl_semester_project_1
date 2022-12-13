import math
import numpy as np
from unicodedata import name
import pyLasaDataset as lasa
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles


class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        attractor_positions,
        path,
        start_position=np.array([0.0, 0.0]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.attractor_positions = attractor_positions
        self.counter = 0
        self.path = path

        self.dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.obstacle_environment,
        )

        self.position_list = np.zeros((self.dim, self.it_max))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        # Here come the main calculation part
        velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii - 1])

        self.position_list[:, ii] = (
            velocity * self.dt_simulation + self.position_list[:, ii - 1]
        )

        dist = self.position_list[:, ii - 1] - \
            self.initial_dynamics.attractor_position
        if (math.sqrt(dist[0] * dist[0] + dist[1] * dist[1]) < 0.1):
            if (self.counter < self.attractor_positions.shape[0]):
                self.initial_dynamics.attractor_position = self.attractor_positions[
                    self.counter, :]
                self.counter += 1

        # print(

        # Update obstacles
        self.obstacle_environment.do_velocity_step(
            delta_time=self.dt_simulation)

        self.ax.clear()

        self.ax.plot(self.path[:, 0], self.path[:, 1], markersize=0.2,
                     marker=".", color="cyan")
        for pos in self.attractor_positions:
            self.ax.scatter(pos[0], pos[1], marker="x", color="red")

        # Drawing and adjusting of the axis
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )
        self.ax.plot(
            self.position_list[0, ii],
            self.position_list[1, ii],
            "o",
            color="#135e08",
            markersize=12,
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

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        # np.allclose(self.position_list[:, ii], self.position_list[:, ii - 1])
        return np.allclose(self.position_list[:, ii], self.attractor_positions[-1])


def compute_attractor_positions_with_gmm(index, gmm_n):
    data = lasa.DataSet.BendedLine.demos[index]
    pos = data.pos.transpose()
    time = data.t.transpose()
    X = np.hstack((pos, time))
    gm = GaussianMixture(n_components=gmm_n, random_state=0).fit(X)
    gmm_means = gm.means_
    return gmm_means[gmm_means[:, 2].argsort()][:, 0:2]


def run_path_following_robot():
    id = 0
    obstacle_environment = ObstacleContainer()
    attractor_positions = compute_attractor_positions_with_gmm(id, 7)
    path = lasa.DataSet.BendedLine.demos[id].pos.transpose()
    initial_position = np.array([-10.5, -2.0])
    initial_dynamics = LinearSystem(
        attractor_position=initial_position,
        maximum_velocity=5.0,
        distance_decrease=0.3,
    )

    my_animation = DynamicalSystemAnimation(
        it_max=2000,
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        start_position=initial_dynamics.attractor_position,
        attractor_positions=attractor_positions,
        path=path,
        x_lim=[-42.0, 2.0],
        y_lim=[-4.0, 12.0],
    )

    my_animation.run(save_animation=False)


if __name__ == '__main__':
    run_path_following_robot()

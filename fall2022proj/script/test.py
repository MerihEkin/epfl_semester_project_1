import matlab.engine
import numpy as np

import pyLasaDataset as lasa
import matplotlib.pyplot as plt
from matplotlib import animation

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

import lpv_ds


class DSAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        attractor_dynamics,
        obstacle_environment,
        obstacle_targets,
        reference_path,
        start_position=np.array([0.0, 0.0]),
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
    ):
        self.counter = 0
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.initial_dynamics = initial_dynamics
        self.attractor_dynamics = attractor_dynamics
        self.obstacle_environment = obstacle_environment
        self.obstacle_targets = obstacle_targets
        self.path = reference_path

        assert (len(self.obstacle_environment._obstacle_list)
                == len(self.obstacle_targets))

        self.agent_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.obstacle_environment
        )  # for the agent
        self.agent_position = np.zeros((self.dim, self.it_max))
        self.agent_position[:, 0] = start_position

        self.attractor_avoiders = []  # for attractor positions
        self.attractor_positions = []

        for i in range(0, len(self.attractor_dynamics)):
            self.attractor_avoiders.append(ModulationAvoider(
                initial_dynamics=self.attractor_dynamics[i], obstacle_environment=self.obstacle_environment))
            attractor_pos = np.zeros((self.dim, self.it_max))
            attractor_pos[:, 0] = self.attractor_dynamics[i].attractor_position
            self.attractor_positions.append(attractor_pos)

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        # update obstacle velocities
        for i in range(len(self.obstacle_targets)):
            target_pos = self.obstacle_targets[i]
            current_pos = self.obstacle_environment._obstacle_list[i].center_position
            dst = target_pos - current_pos
            dst_norm = np.linalg.norm(dst, 2)
            if (np.linalg.norm(dst, 2) < 0.1 and not np.allclose(self.obstacle_environment._obstacle_list[i].linear_velocity, np.array([0.0, 0.0]))):
                self.obstacle_environment._obstacle_list[i].linear_velocity = np.array([
                    0.0, 0.0])
            elif (dst_norm < 0.1):
                pass
            else:
                self.obstacle_environment._obstacle_list[i].linear_velocity = dst / dst_norm

        # update obstacle positions
        self.obstacle_environment.do_velocity_step(
            delta_time=self.dt_simulation)

        # update attractor positions
        for i in range(0, len(self.attractor_dynamics)):
            velocity = self.attractor_avoiders[i].evaluate(
                self.attractor_positions[i][:, ii - 1])

            self.attractor_positions[i][:, ii] = (
                velocity * self.dt_simulation +
                self.attractor_positions[i][:, ii - 1]
            )

        # update current attractor
        if (self.counter < len(self.attractor_positions)):
            self.initial_dynamics.attractor_position = self.attractor_positions[
                self.counter][:, ii]
            dst = self.initial_dynamics.attractor_position - \
                self.agent_position[:, ii-1]
            dst_norm = np.linalg.norm(dst, 2)
            if (dst_norm < 0.2):
                self.counter += 1

        # update agent position
        velocity = self.agent_avoider.evaluate(
            self.agent_position[:, ii - 1])
        self.agent_position[:, ii] = (
            velocity * self.dt_simulation + self.agent_position[:, ii-1])

        # draw current simulation
        self.ax.clear()
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        self.ax.plot(self.path[:, 0], self.path[:, 1], markersize=0.25,
                     marker=".", color="yellowgreen")

        self.ax.plot(
            self.agent_position[0, :ii], self.agent_position[1, :ii], ":", color="#135e08"
        )

        for i in range(0, len(self.attractor_dynamics)):
            # Drawing and adjusting of the axis
            self.ax.plot(
                self.attractor_positions[i][0, :ii], self.attractor_positions[i][1, :ii], ":", color="#135e08"
            )
            self.ax.plot(
                self.attractor_positions[i][0, ii],
                self.attractor_positions[i][1, ii],
                "k*",
                markersize=8,
            )

        self.ax.plot(
            self.agent_position[0, ii],
            self.agent_position[1, ii],
            "o",
            color="#135e08",
            markersize=12,
        )

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            color='r',
            markersize=8,
        )

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
        # if self.counter != 0 and np.allclose(self.obstacle_environment._obstacle_list[0].linear_velocity.all(), np.array([0.0, 0.0])):
        #     for ps in self.position_list:
        #         if not np.allclose(ps[:, ii], ps[:, ii-1]):
        #             return False
        #         return True
        return False


def compute_attractor_positions_with_gmm(index, gmm_n):
    data = lasa.DataSet.BendedLine.demos[index]
    pos = data.pos.transpose()
    time = data.t.transpose()
    X = np.hstack((pos, time))
    gm = GaussianMixture(n_components=gmm_n, random_state=0).fit(X)
    gmm_means = gm.means_
    return gmm_means[gmm_means[:, 2].argsort()][:, 0:2]


def get_gmm_from_matlab(position, velocity):
    est_options = {
        # "type": "diag",
        "type": 0,
        "maxK": 15.0,
        "fixed_K": matlab.double([]),
        "samplerIter": 20.0,
        "do_plots": 0,
        "sub_sample": 1,
        "estimate_l": 1.0,
        "l_sensitivity": 2.0,
        "length_scale": matlab.double([]),
    }

    start_global_matlab_engine = True

    if start_global_matlab_engine and not "matlab_eng" in locals():
        matlab_eng = matlab.engine.start_matlab()

    data_py = np.vstack((position, velocity))

    pos_array = matlab.double(position.astype('float64'))
    vel_array = matlab.double(velocity.astype('float64'))
    Data = matlab.double(data_py.astype('float64'))

    priors, mu, sigma = matlab_eng.fit_gmm(
        pos_array, vel_array, est_options, nargout=3)

    lyap_constr = 2

    ds_gmm = {
        "Mu": mu,
        "Priors": priors,
        "Sigma": sigma,
    }

    init_cvx = 1
    symm_constr = 0

    Vxf = matlab_eng.learn_wsaqf(Data, nargout=1)
    P_opt = Vxf["P"]

    att_py = np.array([0.0, 0.0])
    att = matlab.double(att_py.astype('float64'))
    att = matlab_eng.transpose(att)

    A_k, b_k, P = matlab_eng.optimize_lpv_ds_from_data(
        Data, att, lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr, nargout=3)

    matlab_eng.quit()

    priors = np.array(priors)
    mu = np.array(mu)
    sigma = np.array(sigma)
    A_k = np.array(A_k)
    b_k = np.array(b_k)
    P = np.array(P)

    return priors, mu, sigma, A_k, b_k, P


def run_simulation():

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[2.0, 2.0],
            center_position=np.array([-5.0, 10.0]),
            margin_absolut=0.5,
            orientation=0,
            linear_velocity=np.array([0.0, 0.0]),
            tail_effect=False,
        )
    )
    obstacle_environment.append(
        Ellipse(
            axes_length=[2.0, 2.0],
            center_position=np.array([-30.0, 4.0]),
            margin_absolut=0.5,
            orientation=0,
            linear_velocity=np.array([0.0, 0.0]),
            tail_effect=False,
        )
    )

    path_index = 0
    reference_path = lasa.DataSet.BendedLine.demos[path_index].pos.transpose()

    attractor_dynamics = []
    gmm_centorids = compute_attractor_positions_with_gmm(path_index, 7)
    data = lasa.DataSet.BendedLine.demos[path_index]
    pos = data.pos
    vel = data.vel
    priors, mu, sigma, A_k, b_k, P = get_gmm_from_matlab(pos, vel)

    A_g = np.array(A_k)
    b_g = np.array(b_k)
    gmmvar = lpv_ds.GmmVariables(mu, priors, sigma)
    x = np.array([0, 0])
    x.shape = (2, 1)
    lpvds = lpv_ds.LpvDs(1e-5, 2e-200)
    x_dot = lpvds.evaluate(x, gmmvar, A_g, b_g)

    gmm_centorids = np.transpose(b_k)

    for i in range(len(gmm_centorids)):
        attractor_dynamics.append(
            LinearSystem(
                attractor_position=gmm_centorids[i],
                maximum_velocity=1.5,
                distance_decrease=0.3,
            )
        )

    initial_dynamics = LinearSystem(
        attractor_position=gmm_centorids[0], maximum_velocity=1.5, distance_decrease=0.3,)

    my_animation = DSAnimation(
        it_max=10000,
        dt_simulation=0.05,
        dt_sleep=0.01,
    )

    obstacle_targets = []
    obstacle_targets.append(gmm_centorids[5])
    obstacle_targets.append(gmm_centorids[2])

    my_animation.setup(initial_dynamics=initial_dynamics,
                       attractor_dynamics=attractor_dynamics,
                       obstacle_environment=obstacle_environment,
                       obstacle_targets=obstacle_targets,
                       reference_path=reference_path,
                       start_position=np.array(
                           [reference_path[0, 0], reference_path[0, 1]]),
                       x_lim=[-42.0, 2.0],
                       y_lim=[-4.0, 12.0],)

    my_animation.run(save_animation=False)


if (__name__) == "__main__":

    run_simulation()

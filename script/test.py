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
            eps=eps, realmin=realmin, algorithm=lpv_ds.Algorithm.Weighted3)

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
        elif self.agent_dynamics.algorithm == lpv_ds.Algorithm.Weighted1 or self.agent_dynamics.algorithm == lpv_ds.Algorithm.Weighted2 \
                or self.agent_dynamics.algorithm == lpv_ds.Algorithm.Weighted3:
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
        self.ax.legend()
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
            self.agent_positions[0, :ii], self.agent_positions[1, :ii], linewidth='3.0', color="#135e08", label="Computed Trajectory")
        self.ax.plot(
            self.agent_positions[0, ii], self.agent_positions[1, ii], "o", color="#135e08", markersize=12,)

    def has_converged(self, ii) -> bool:
        return False


def get_gmm_from_matlab(position, velocity, final_pos: np.array):

    lyap_constr = 2
    init_cvx = 1
    symm_constr = 0

    est_options = {
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

    data_py = np.vstack((position, velocity))

    if not "matlab_eng" in locals():
        matlab_eng = matlab.engine.start_matlab()

    pos_array = matlab.double(position.astype('float64'))
    vel_array = matlab.double(velocity.astype('float64'))
    Data = matlab.double(data_py.astype('float64'))

    priors, mu, sigma = matlab_eng.fit_gmm(
        pos_array, vel_array, est_options, nargout=3)

    ds_gmm = {
        "Mu": mu,
        "Priors": priors,
        "Sigma": sigma,
    }

    Vxf = matlab_eng.learn_wsaqf(Data, nargout=1)
    P_opt = Vxf["P"]

    att = matlab.double(final_pos.astype('float64'))
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


def get_Ak_and_bk_from_gmm(position, velocity, final_pos: np.array, priors, mu, sigma):

    lyap_constr = 2
    init_cvx = 1
    symm_constr = 0

    est_options = {
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

    data_py = np.vstack((position, velocity))

    if not "matlab_eng" in locals():
        matlab_eng = matlab.engine.start_matlab()

    pos_array = matlab.double(position.astype('float64'))
    vel_array = matlab.double(velocity.astype('float64'))
    Data = matlab.double(data_py.astype('float64'))

    priors = matlab.double(priors.astype('float64'))
    mu = matlab.double(mu.astype('float64'))
    sigma = matlab.double(sigma.astype('float64'))

    ds_gmm = {
        "Mu": mu,
        "Priors": priors,
        "Sigma": sigma,
    }

    Vxf = matlab_eng.learn_wsaqf(Data, nargout=1)
    P_opt = Vxf["P"]

    att = matlab.double(final_pos.astype('float64'))
    att = matlab_eng.transpose(att)

    A_k, b_k, P = matlab_eng.optimize_lpv_ds_from_data(
        Data, att, lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr, nargout=3)

    matlab_eng.quit()

    A_k = np.array(A_k)
    b_k = np.array(b_k)
    P = np.array(P)

    return A_k, b_k, P


def run_simulation():
    path_index = 0
    path = lasa.DataSet.LShape.demos[path_index]
    pos = path.pos
    vel = path.vel
    final_pos = pos[:, -1]

    # x_lim = [-50.0, 2.0] # sine
    # y_lim = [-9.0, 17.0] # sine
    x_lim = [-36.0, 3.0]  # LShape
    y_lim = [-13.0, 48.0]  # LShape

    # priors, mu, sigma, A_k, b_k, _ = get_gmm_from_matlab(pos, vel, final_pos)

    priors = np.array([[0.164, 0.211, 0.249, 0.165, 0.211]])
    mu = np.array([[-27.97390462, -29.03648456,  -6.1789614, -24.65764882,
                   -31.10150071],
                   [19.59989514,  39.13652236,  -0.7555651,  -2.28375025,
                    4.54848509]])
    sigma = np.array([[[1.03796892,  1.18131439, 27.08671255, 23.01309662,
                        3.55017427],
                       [1.13976659, -4.22530448,  2.39282625,  0.75962335,
                        7.0116795]],

                      [[1.13976659, -4.22530448,  2.39282625,  0.75962335,
                        7.0116795],
                       [23.72071442, 27.56720204,  2.06683289,  3.2512121,
                          20.04156201]]])
    A_k = np.array([[[-0.14052669,  -0.78031162,  -0.26195104,  -0.5259253,
                      0.04979977],
                     [-0.15149797,  -0.49104621, -21.92820964,  -6.21825116,
                      -1.07633619]],

                    [[-0.15149797,   0.27951289,   0.57422758,   0.69877895,
                      0.41471347],
                     [-1.91330532,  -0.27261989,  -7.22325673,  -6.59743802,
                        -1.53462801]]])
    b_k = np.array([[-5.05414989e-10, -2.33293689e-11,  1.65762129e-09,
                     -4.49657373e-11,  1.32345018e-11],
                    [1.17332605e-11, -5.62835898e-10,  2.36063974e-10,
                     1.67950213e-11, -1.80756117e-11]])  # Lshape

    # priors = np.array([[0.157, 0.141, 0.151, 0.141, 0.115, 0.148, 0.147]])
    # mu = np.array([[-44.32796508,  -7.79043093, -15.25844007, -37.90240443,
    #                -25.26509153, -29.4662016,  -2.25582531],
    #                [2.59941578,   7.44358783,  -1.48816821,  13.6131425,
    #                 -4.65889144,   6.37242324,   3.66078997]])
    # sigma = np.array([[[5.42808318,  7.74141396,  9.88398331, 10.64780976,
    #                     6.97055789,  4.64377947,  7.5294298],
    #                    [3.14698543,  1.21452297,  9.87384938,  1.34389145,
    #                     -3.48700046, -6.93117754, -3.72495601]],

    #                   [[3.14698543,  1.21452297,  9.87384938,  1.34389145,
    #                     -3.48700046, -6.93117754, -3.72495601],
    #                    [11.16031726,  0.99469511, 14.05689984,  2.72896659,
    #                       5.29744191, 19.12354504,  6.63244693]]])
    # A_k = np.array([[[-0.15361387,  -0.29272617,  -0.91868153,   0.20439437,
    #                   -0.26925165,  -0.21949873, -11.2734523],
    #                  [-0.22670164,   0.57651046,   0.06924674,   1.54493723,
    #                   -0.90117649,   0.05772265,  -4.73276]],

    #                 [[-0.22670164,  -2.0260507,  -1.01839015,  -1.72820046,
    #                   0.79005768,   0.58074487,   5.91981563],
    #                  [-0.42940383,  -1.36539948,  -0.16480172,  -4.40107783,
    #                     -1.86535077,  -0.26854294,   0.69763456]]])
    # b_k = np.array([[-1.40084397e-10, -1.08401671e-11, -2.42144074e-11,
    #                 -1.41860819e-12, -1.39397708e-11, -2.18533741e-11,
    #                 8.60399450e-12],
    #                 [1.74945861e-10,  8.28287095e-12,  5.09425721e-11,
    #                  7.81775787e-13, -3.51631037e-12, -9.64815119e-12,
    #                  -8.33311784e-11]]) # Sine

    ds_gmm = lpv_ds.GmmVariables(mu, priors, sigma)

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[5.0, 15.0],
            # center_position=np.array([-12.0, 28.0]),
            center_position=np.array([-19.9, -2.47]),
            margin_absolut=0.5,
            tail_effect=False,
        )
    )

    # obstacle_environment.append(
    #     Ellipse(
    #         axes_length=[12.0, 12.0],
    #         center_position=np.array([-29.47, 6.33]),
    #         margin_absolut=0.5,
    #         orientation=-45*pi/180,
    #         tail_effect=False,
    #     )
    # ) # Sine

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
        # start_position=np.array([-44.63, -0.9]), # sine
        start_position=np.array([-29.4, 43.2]),  # L Shape
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

#!/usr/bin/env python3
from numpy import genfromtxt
from rclpy.node import Node
from network_interfaces.zmq.network import CommandMessage
from network_interfaces.control_type import ControlType
from franka_avoidance.robot_interface import \
    RobotZmqInterface as RobotInterface
from dynamical_systems import DYNAMICAL_SYSTEM_TYPE, create_cartesian_ds
from controllers import CONTROLLER_TYPE, create_cartesian_controller
import state_representation as sr

from franka_avoidance.optitrack_container import OptitrackContainer
from franka_avoidance.pybullet_handler import PybulletHandler
from franka_avoidance.rviz_handler import RvizHandler
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.containers import ObstacleContainer

import threading

# import script.lpv_ds as lpvds
# from .script import lpv_ds as lpvds
# import lpv_ds as lpvds
# import lpv_ds as lpvds
# from initial_dynamics import InitialDynamics, InitialDynamicsType

import learning_avoidance.lpv_ds as lpvds
from learning_avoidance.initial_dynamics import InitialDynamics, InitialDynamicsType

import numpy as np
import rclpy
import sys

# Custom libraries

data_path = '/home/ros2/ros2_ws/src/franka_avoidance/project_ekin/data/'


class TwistController(Node):
    def __init__(self, robot, freq: float = 100, node_name="twist_controller", is_simulation: bool = True):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        self.command = CommandMessage()
        self.command.control_type = [ControlType.EFFORT.value]

        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        self.ds.set_parameter_value(
            # "gain", [200.0, 200.0, 200.0, 50.0, 50.0, 50.0],
            # "gain", [1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0],
            "gain", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            sr.ParameterType.DOUBLE_ARRAY
        )

        self.ctrl = create_cartesian_controller(
            CONTROLLER_TYPE.COMPLIANT_TWIST)
        if is_simulation:
            self.ctrl.set_parameter_value(
                "linear_principle_damping", 1., sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "linear_orthogonal_damping", 1., sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "angular_stiffness", .5, sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "angular_damping", .5, sr.ParameterType.DOUBLE)
        else:
            self.ctrl.set_parameter_value(
                "linear_principle_damping", 50., sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "linear_orthogonal_damping", 50., sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "angular_stiffness", 2., sr.ParameterType.DOUBLE)
            self.ctrl.set_parameter_value(
                "angular_damping", 2., sr.ParameterType.DOUBLE)

        experiment_nr = 10
        priors = genfromtxt(
            data_path + f'priors{experiment_nr}.csv', delimiter=',')
        K = priors.size
        priors.shape = (1, K)
        mus = genfromtxt(data_path+f'mus{experiment_nr}.csv', delimiter=',')
        sigmas = genfromtxt(
            data_path+f'sigmas{experiment_nr}.csv', delimiter=',')
        sigmas = np.reshape(sigmas, (sigmas.shape[0], sigmas.shape[0], -1))
        A_k = genfromtxt(data_path+f'A_k{experiment_nr}.csv', delimiter=',')
        A_k = np.reshape(A_k, (A_k.shape[0], A_k.shape[0], -1))
        b_k = genfromtxt(data_path+f'b_k{experiment_nr}.csv', delimiter=',')
        P = genfromtxt(data_path+f'P{experiment_nr}.csv', delimiter=',')
        gmm_ds = lpvds.GmmVariables(mu=mus, priors=priors, sigma=sigmas)
        self.lpv_ds = lpvds.LpvDs(A_k=A_k, b_k=b_k, ds_gmm=gmm_ds)
        # obstacle_environment = ObstacleContainer()
        self.obstacles = OptitrackContainer(use_optitrack=True)
        self.obstacles.append(
            Ellipse(
                center_position=np.array([0.0, 0.0, 0.0]),
                axes_length=np.array([0.1, 0.06, 0.16]),
                margin_absolut=0.16,
                linear_velocity=np.zeros(3),
                # tail_effect=False,
            ),
            obstacle_id=28,
        )
        # self.obstacles.visualization_handler = PybulletHandler(self.obstacles)
        self.obstacles.visualization_handler = RvizHandler(self.obstacles)
        self.initial_dynamics = InitialDynamics(
            maximum_velocity=1.0,
            attractor_position=np.array(
                [0.40939928740376075, -0.6546517367820528, 0.22282657197812547]),
            dimension=A_k.shape[0],
        )
        self.initial_dynamics.setup(trajectory_dynamics=self.lpv_ds,
                                    a=100,
                                    b=50,
                                    obstacle_environment=self.obstacles,
                                    initial_dynamics_type=InitialDynamicsType.WeightedSum,
                                    )

        self.dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.obstacles,
        )

    def run(self):
        target_set = False

        while rclpy.ok():
            state = self.robot.get_state()

            self.obstacles.update()
            # print("Updated obstacles.")

            if not state:
                continue
            if not target_set:
                target = sr.CartesianPose(
                    state.ee_state.get_name(),
                    np.array([0.3, 0.4, 0.5]),
                    np.array([0.0, 1.0, 0.0, 0.0]),
                    state.ee_state.get_reference_frame(),
                )
                self.ds.set_parameter_value(
                    "attractor",
                    target,
                    sr.ParameterType.STATE,
                    sr.StateType.CARTESIAN_POSE,
                )
                target_set = True
            else:
                current_state = sr.CartesianState(state.ee_state)
                position = current_state.get_position()

                # x = np.array([position[0], position[1], position[2]])
                x = np.array([position])
                x_dot = self.dynamic_avoider.evaluate(position=x)

                # print(np.round(x_dot, 4))
                max_velocity = 0.25
                if np.linalg.norm(x_dot) > max_velocity:
                    x_dot = max_velocity * x_dot / np.linalg.norm(x_dot)

                # print(np.round(x_dot, 4))
                cartesian_state = self.ds.evaluate(state.ee_state)
                cartesian_state.set_linear_velocity(x_dot)

                # twist = sr.CartesianTwist(self.ds.evaluate(state.ee_state))
                twist = sr.CartesianTwist(cartesian_state)
                twist.clamp(0.25, 0.5)
                print(twist)
                self.command_torques = sr.JointTorques(
                    self.ctrl.compute_command(
                        twist, state.ee_state, state.jacobian)
                )
                self.command.joint_state = state.joint_state
                self.command.joint_state.set_torques(
                    self.command_torques.get_torques())

                self.robot.send_command(self.command)
                # print(self.command.joint_state.get_torques())

            self.rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    # rospy.init_node("test", anonymous=True)
    robot_interface = RobotInterface("*:1601", "*:1602")

    # Spin in a separate thread
    controller = TwistController(
        robot=robot_interface, freq=500, is_simulation=False)

    thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True)
    thread.start()

    try:
        controller.run()

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    thread.join()

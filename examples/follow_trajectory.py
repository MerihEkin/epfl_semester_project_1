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

from dynamic_obstacle_avoidance.containers import ObstacleContainer
import threading

# import script.lpv_ds as lpvds
# from .script import lpv_ds as lpvds
# import lpv_ds as lpvds
# import lpv_ds as lpvds
# from initial_dynamics import InitialDynamics, InitialDynamicsType

import src.lpv_ds as lpvds
from src.initial_dynamics import InitialDynamics, InitialDynamicsType

import numpy as np
import rclpy
import sys

# Custom libraries

data_path = '/home/ros2/ros2_ws/src/franka_obstacle_avoidance/project_ekin/fall2022proj/data/'


class TwistController(Node):
    def __init__(self, robot, freq: float = 100, node_name="twist_controller"):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        self.command = CommandMessage()
        self.command.control_type = [ControlType.EFFORT.value]

        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        self.ds.set_parameter_value(
            "gain", [50.0, 50.0, 50.0, 10.0, 10.0,
                     10.0], sr.ParameterType.DOUBLE_ARRAY
        )

        self.ctrl = create_cartesian_controller(
            CONTROLLER_TYPE.COMPLIANT_TWIST)
        self.ctrl.set_parameter_value(
            "linear_principle_damping", 1.0, sr.ParameterType.DOUBLE
        )
        self.ctrl.set_parameter_value(
            "linear_orthogonal_damping", 1.0, sr.ParameterType.DOUBLE
        )
        self.ctrl.set_parameter_value(
            "angular_stiffness", 0.5, sr.ParameterType.DOUBLE)
        self.ctrl.set_parameter_value(
            "angular_damping", 0.5, sr.ParameterType.DOUBLE)

        priors = genfromtxt(data_path+'priors.csv', delimiter=',')
        K = priors.size
        priors.shape = (1, K)
        mus = genfromtxt(data_path+'mus.csv', delimiter=',')
        sigmas = genfromtxt(data_path+'sigmas.csv', delimiter=',')
        sigmas = np.reshape(sigmas, (sigmas.shape[0], sigmas.shape[0], -1))
        A_k = genfromtxt(data_path+'A_k.csv', delimiter=',')
        A_k = np.reshape(A_k, (A_k.shape[0], A_k.shape[0], -1))
        b_k = genfromtxt(data_path+'b_k.csv', delimiter=',')
        P = genfromtxt(data_path+'P.csv', delimiter=',')
        gmm_ds = lpvds.GmmVariables(mu=mus, priors=priors, sigma=sigmas)
        self.lpv_ds = lpvds.LpvDs(A_k=A_k, b_k=b_k, ds_gmm=gmm_ds)
        obstacle_environment = ObstacleContainer()
        self.initial_dynamics = InitialDynamics(
            maximum_velocity=1.0,
            attractor_position=np.array(
                [0.3756374418735504, -0.24511094391345978, 0.3886714279651642]),
            dimension=2,
        )
        self.initial_dynamics.setup(trajectory_dynamics=lpvds,
                                    a=100,
                                    b=50,
                                    obstacle_environment=obstacle_environment,
                                    initial_dynamics_type=InitialDynamicsType.LocallyRotatedFromObstacle,
                                    )

    def run(self):
        target_set = False

        while rclpy.ok():
            state = self.robot.get_state()
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
                cartesian_state = sr.CartesianState(state.ee_state)
                position = cartesian_state.get_position()
                # x = np.array([position[0], position[1], position[2]])
                x = np.array([position])
                x_dot = self.initial_dynamics.evaluate(x=x)
                cartesian_state.set_linear_velocity(x_dot)
                twist = sr.CartesianTwist(cartesian_state)
                twist.clamp(0.25, 0.5)
                self.command_torques = sr.JointTorques(
                    self.ctrl.compute_command(
                        twist, state.ee_state, state.jacobian)
                )
                self.command.joint_state = state.joint_state
                self.command.joint_state.set_torques(
                    self.command_torques.get_torques())

                self.robot.send_command(self.command)

            self.rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    # rospy.init_node("test", anonymous=True)
    robot_interface = RobotInterface("*:1601", "*:1602")

    # Spin in a separate thread
    controller = TwistController(robot=robot_interface, freq=500)

    thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True)
    thread.start()

    try:
        controller.run()

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    thread.join()

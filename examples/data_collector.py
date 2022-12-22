#!/usr/bin/env python3
import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node

import threading
import signal

import time
# LASA Libraries
import state_representation as sr
from dynamical_systems import create_cartesian_ds, DYNAMICAL_SYSTEM_TYPE
from controllers import create_cartesian_controller, CONTROLLER_TYPE
from network_interfaces.control_type import ControlType
from network_interfaces.zmq.network import CommandMessage

# Custom libraries
from franka_avoidance.robot_interface import RobotZmqInterface as RobotInterface

# data_path = '/home/ros2/ros2_ws/src/franka_obstacle_avoidance/project_ekin/data/'
data_path = '/home/ros2/ros2_ws/src/franka_avoidance/project_ekin/data/'


class JointSpaceController(Node):
    def __init__(self, robot, freq: float = 100, node_name="joint_controller", max_iter=2000):
        super().__init__(node_name)
        self.robot = robot
        self.rate = self.create_rate(freq)

        signal.signal(signal.SIGINT, self.control_c_handler)

        self.max_iter = max_iter
        self.trajectory = np.zeros((max_iter, 19))
        self.current_iter = 0
        self.delta_time = 1.0 / freq

        self.command = CommandMessage()
        self.command.control_type = [ControlType.VELOCITY.value]

        self.ds = create_cartesian_ds(DYNAMICAL_SYSTEM_TYPE.POINT_ATTRACTOR)
        self.ds.set_parameter_value(
            "gain", [50.0, 50.0, 50.0, 10.0, 10.0,
                     10.0], sr.ParameterType.DOUBLE_ARRAY
        )

        self.ctrl = create_cartesian_controller(
            CONTROLLER_TYPE.COMPLIANT_TWIST)
        self.ctrl.set_parameter_value(
            "linear_principle_damping", 0., sr.ParameterType.DOUBLE)
        self.ctrl.set_parameter_value(
            "linear_orthogonal_damping", 0., sr.ParameterType.DOUBLE)
        self.ctrl.set_parameter_value(
            "angular_stiffness", 0.01, sr.ParameterType.DOUBLE)
        self.ctrl.set_parameter_value(
            "angular_damping", 0.01, sr.ParameterType.DOUBLE)

    def save_data(self):
        arr = np.array(self.trajectory[0:self.max_iter, :], copy=True)
        df = pd.DataFrame(self.trajectory, columns=[
            'position', 'position', 'position',
            'orientation', 'orientation', 'orientation', 'orientation',
            'linear_velocity', 'linear_velocity', 'linear_velocity',
            'angular_velocity', 'angular_velocity', 'angular_velocity',
            'linear_acceleration', 'linear_acceleration', 'linear_acceleration',
            'angular_acceleration', 'angular_acceleration', 'angular_acceleration'
        ])
        df.to_csv(data_path + 'trajectory3.csv',
                  index=False, header=True, encoding='utf-8', sep=',')
        print(f'\niter: {self.current_iter}')
        print("Trajector is saved!!")

    def control_c_handler(self, sig, frame):
        self.save_data()
        rclpy.shutdown()

    def run(self):

        while rclpy.ok():
            state = self.robot.get_state()
            if not state:
                # breakpoint()
                time.sleep(self.delta_time)
                continue
            cartesian_state = sr.CartesianState(state.ee_state)
            oo = cartesian_state.get_orientation()

            target = sr.CartesianPose(
                state.ee_state.get_name(),
                cartesian_state.get_position(),
                np.array([oo[0], oo[1], oo[2], oo[3]]),
                state.ee_state.get_reference_frame()
            )

            self.ds.set_parameter_value(
                "attractor", target, sr.ParameterType.STATE, sr.StateType.CARTESIAN_POSE)

            twist = sr.CartesianTwist(self.ds.evaluate(state.ee_state))
            twist.clamp(.01, .01)
            # print("Desired Angular Velocities: ", twist.data()[3:6])
            command_torques = sr.JointTorques(
                self.ctrl.compute_command(twist, state.ee_state, state.jacobian))
            self.command.joint_state = state.joint_state
            # print("Command:", command.joint_state.get_torques())
            # self.command.joint_state.set_torques(command_torques.get_torques())

            # self.robot.send_command(self.command)

            position = cartesian_state.get_position()
            orientation = cartesian_state.get_orientation()
            linear_velocity = cartesian_state.get_linear_velocity()
            angular_velocity = cartesian_state.get_angular_velocity()
            linear_acceleration = cartesian_state.get_linear_acceleration()
            angular_acceleration = cartesian_state.get_angular_acceleration()
            self.current_iter += 1
            if self.current_iter < self.max_iter:
                self.trajectory[self.current_iter, :] = np.array(
                    [position[0],
                     position[1],
                     position[2],
                     orientation[0],
                     orientation[1],
                     orientation[2],
                     orientation[3],
                     linear_velocity[0],
                     linear_velocity[1],
                     linear_velocity[2],
                     angular_velocity[0],
                     angular_velocity[1],
                     angular_velocity[2],
                     linear_acceleration[0],
                     linear_acceleration[1],
                     linear_acceleration[2],
                     angular_acceleration[0],
                     angular_acceleration[1],
                     angular_acceleration[2]])

            elif self.current_iter == self.max_iter:
                print("Maximum number of iterations for data collection is reached!")
            time.sleep(self.delta_time)


if __name__ == "__main__":
    rclpy.init()
    # rospy.init_node("test", anonymous=True)
    robot_interface = RobotInterface("*:1601", "*:1602")

    # Spin in a separate thread
    controller = JointSpaceController(
        robot=robot_interface, freq=50, max_iter=1500)

    thread = threading.Thread(
        target=rclpy.spin, args=(controller,), daemon=True)
    thread.start()

    # try:
    #     controller.run()

    # except KeyboardInterrupt:
    #     controller.save_data()
    # rclpy.shutdown()

    controller.run()

    thread.join()

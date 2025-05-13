import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig

import os

# TODO: we probably want a common package to hold this, not copy and pasted. This is just for prototyping for now.
from common.contact_aided_invariant_ekf.contact_aided_invariant_ekf import (
    RightInEKF,
)
from common.contact_aided_invariant_ekf.robot_state import RobotState
from common.contact_aided_invariant_ekf.noise_params import NoiseParams
from common.contact_aided_invariant_ekf.drake_kinematics import DrakeKinematics

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        urdf_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        self.hwi = HWI(self.duck_config, serial_port)

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused

        self.command_freq = 20  # hz
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # Reference motion, but we only really need the length of one phase
        # TODO
        self.PRM = PolyReferenceMotion("./polynomial_coefficients.pkl")
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()
        # TODO: InEKF, lots of settings should be from config. Prototyping for now.

        # Generic origin for the IMU when the duck is at initial position.
        initial_pose = np.array([[1, 0,  0, -0.08],
                                [0,  1,  0,  0],
                                [0,  0,  1,  0.27],
                                [0,  0,  0,  1]])

        # Convert to rotation matrix
        body_rotation_matrix = initial_pose[0:3, 0:3]
        self.ekf_noise_params = NoiseParams() # TODO: put in real
        # self.ekf_noise_params.set_gyroscope_bias_noise(1e-6)
        # ekf_noise_params.set_contact_noise(0.5)
        ekf_robot_state = RobotState()
        ekf_robot_state.set_position(initial_pose[0:3, 3])
        ekf_robot_state.set_rotation(body_rotation_matrix)
        ekf_robot_state.set_velocity(np.zeros(3))
        self._ekf = RightInEKF(ekf_robot_state, self.ekf_noise_params)

        # Define the forward kinematics source.
        self.drake_kinematics = DrakeKinematics(
            imu_frame_name="imu",
            urdf_model_path=urdf_model_path,
            end_effector_frame_name_to_id_map={
                "foot_assembly": 0,
                "foot_assembly_2": 1,
            },
            # From URDF assembly to foot. And then add additional padding from foot link to bottom of foot.
            end_effector_offset_map={
                "foot_assembly": [0.0005, -0.036225 - 0.008, 0.01955],
                "foot_assembly_2": [0.0005, -0.036225 - 0.008, 0.01955],
            },
        )

        self.mask = np.zeros(
            len(self.drake_kinematics.joint_name_to_idx_map), dtype=int
        )
        for name, idx in self.drake_kinematics.joint_name_to_idx_map.items():
            self.mask[idx] = self.hwi.actuator_name_to_idx_map[name]

    def get_obs(self, correct_kin: bool = True):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        dof_pos_corrected = dof_pos - self.init_pos
        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()

        # Apply EKF
        ekf_imu_measurement = np.zeros(6)
        ekf_imu_measurement[0:3] = imu_data["gyro"]
        ekf_imu_measurement[3:6] = imu_data["accelero"]

        self._ekf.propagate(
            self.ekf_imu_measurement_prev,  # use previous measurement.
            dt=self.control_freq,
        )

        # Update prev sim time
        self.ekf_imu_measurement_prev = ekf_imu_measurement.copy()

        ekf_contact_measurement = [(idx, feet_contacts[idx]) for idx in range(len(feet_contacts))]
        self._ekf.set_contacts(ekf_contact_measurement)
        # TODO: grab uncertainty from the servos.
        joint_uncertainty = 1e-3
        feet_to_kinematics_map = self.drake_kinematics.compute_kinematics(
            dof_pos_corrected[self.mask],
            joint_uncertainty * np.eye(self.num_dofs),
        )

        if correct_kin:
            self._ekf.correct_kinematics(
                measured_kinematics=list(feet_to_kinematics_map.values())
            )

        estimated_state = self._ekf.get_state()

        rotation_world = estimated_state.get_rotation()
        floating_base_vel_world = estimated_state.get_velocity()

        # TODO: use this when we move to path frame.
        translation_world = estimated_state.get_position()

        # Only use 6D rotation, not full 9D.
        rotation_world_6d = rotation_world[:, 0:2].flatten()
        floating_base_vel_body = rotation_world.T @ floating_base_vel_world[0:3]
        obs = np.concatenate(
            [
                rotation_world_6d,
                floating_base_vel_body,
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos_corrected,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Applying kinematic correction doesn't need to happen every single timestep. Lower the rate if running slowly.
                obs = self.get_obs(
                    correct_kin=True
                )
                if obs is None:
                    continue
                
                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                magic_number = 45 + 6 + 3 # 6d pose, lin vel
                self.obs_history = np.roll(self.obs_history, magic_number)
                self.obs_history[:magic_number] = obs
                obs = self.obs_history

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # action = np.zeros(10)

                self.motor_targets = self.init_pos + action * self.action_scale

                # self.motor_targets = np.clip(
                #     self.motor_targets,
                #     self.prev_motor_targets
                #     - self.max_motor_velocity * (1 / self.control_freq),  # control dt
                #     self.prev_motor_targets
                #     + self.max_motor_velocity * (1 / self.control_freq),  # control dt
                # )

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()

                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)

    parser.add_argument(
        "--urdf_model_path",
        type=str,
        required=True
    )

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        urdf_model_path = args.urdf_model_path
    )
    print("Done instantiating RLWalk")
    rl_walk.run()

from tdmclient import ClientAsync, aw
import numpy as np
import cv2
import numpy as np
from heapq import heappush, heappop


class Thymio:

    async def initiateLock(self):
        self.node = await self.client.wait_for_node()
        await self.node.lock()
        return None

    def __init__(self, l=50, coneMargin=0.1):
        self.client = ClientAsync()
        self.node = None
        self.ratio = 5 / (4003 - 1455)
        self.coneMargin = coneMargin
        # orderThymio = real_speed/speedConversion
        self.speedConversion = 0.797829 / 2.7
        self.sensorAngles = {
            "left_front": -30,
            "front middle-left": -15,
            "front middle": 0,
            "front middle-right": 15,
            "front right": 30,
            "left_back": -135,
            "right_back": 135,
        }

        self.l = l  # mm

        self.L = 46.75  # mm - demi-distance entre les 2 roues
        self.Ts = 0.38
        self.K_rotation = self.L / (8 * self.Ts)
        self.K_translation = 1 / (8 * self.Ts)

        self.W = np.diag([40, 40, 0.1, 40, 40, 0.1])
        self.V_c = np.diag([0.1, 0.1, 0.01, 100, 75.72])
        self.V_nc = 40  # Wheels
        self.A = np.array(
            [
                [1, 0, 0, self.Ts, 0, 0],
                [0, 1, 0, 0, self.Ts, 0],
                [0, 0, 1, 0, 0, self.Ts],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.C_trans = np.array([[0, 1]])

        return None

    def Rotation_theta(self, theta):
        return np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

    def P_1_vers_0(self, theta):
        R_theta = self.Rotation_theta(theta)
        return np.block(
            [
                [R_theta, np.zeros((2, 4))],
                [0, 0, 1, 0, 0, 0],
                [np.zeros((2, 3)), R_theta, np.zeros((2, 1))],
                [np.zeros((1, 6))],
            ]
        )

    def P_0_vers_1(self, theta):
        return self.P_1_vers_0(theta).T

    def g_c(self, z):
        theta = z[2]
        s_c = [
            z[0],  # x
            z[1],  # y
            z[2],  # theta
            z[3] * np.cos(theta)
            + z[4] * np.sin(theta),  # x_dot*cos(theta) + y_dot*sin(theta)
            self.L * z[5],
        ]  # L*theta_dot
        return np.array(s_c)

    def g_nc(self, z):
        return self.C_trans @ z

    def grad_g_c(self, z):
        theta = z[2]
        x_dot, y_dot = z[3], z[4]
        grad = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [
                0,
                0,
                -x_dot * np.sin(theta) + y_dot * np.cos(theta),
                np.cos(theta),
                np.sin(theta),
                0,
            ],
            [0, 0, 0, 0, 0, self.L],
        ]
        return np.array(grad)

    def grad_g_nc(self, z):
        return self.C_trans

    def constructing_s(
        self,
        V_left_measure,
        V_right_measure,
        camera_working=False,
        x_measured=0,
        y_measured=0,
        theta_measured=0,
    ):
        s_nc = (V_left_measure + V_right_measure) / 2
        if camera_working:
            s_c = np.array(
                [
                    x_measured,
                    y_measured,
                    theta_measured,
                    s_nc,
                    (V_left_measure - V_right_measure) / 2,
                ]
            )
            return s_c
        return s_nc

    def getObstaclePosition(self) -> list:
        """
        Calculates the positions of obstacles based on sensor readings.
        The function iterates through 7 sensors, calculates the distance and angle
        for each sensor, and appends the position to a list. If the distance is greater
        than 5, it appends (-1, 0) to indicate no obstacle detected within the threshold.
        Returns:
            list of tuples: A list of tuples where each tuple contains the distance (float) in mm
                            and angle (float) in degrees of the detected obstacle 0 is in front of thymio.
        """
        pos = []
        detected = False
        for i in range(7):
            ##if 2 sensors are close to each other link the 2 dots

            prox = self.getProxH()
            angle = self.sensorAngles[list(self.sensorAngles.keys())[i]]
            distance = self.ratio * (4003 - list(prox)[i])
            if distance > 5:
                pos.append((-1, 0))
            else:
                detected = True
                pos.append((distance * 10, angle))
        return pos, detected

    def filtering_step(
        self,
        z_k_k_1,
        sigma_k_k_1,
        V_left_measure,
        V_right_measure,
        camera_working=False,
        x_measured=0,
        y_measured=0,
        theta_measured=0,
    ):
        ### Computing the variables that are dependant on the state of the camera
        s_k = self.constructing_s(
            V_left_measure,
            V_right_measure,
            camera_working,
            x_measured,
            y_measured,
            theta_measured,
        )
        C_k = self.grad_g_c(z_k_k_1) if camera_working else self.grad_g_nc(z_k_k_1)
        V = self.V_c if camera_working else self.V_nc
        # Changing the computation of the filter based on the state of the camera
        if camera_working:
            g_k = self.g_c(z_k_k_1)
            ### The filtering step that can be rewritten without any problem
            L_k_k = sigma_k_k_1 @ C_k.T @ np.linalg.inv(C_k @ sigma_k_k_1 @ C_k.T + V)
            sigma_k_k = sigma_k_k_1 - L_k_k @ C_k @ sigma_k_k_1
            z_k_k = z_k_k_1 + L_k_k @ (s_k - g_k)

        else:
            ### Changing the frame of coordinates
            theta = z_k_k_1[2]
            z_1_k_k_1 = self.P_1_vers_0(theta) @ z_k_k_1
            sigma_1_k_k_1 = (
                self.P_1_vers_0(theta) @ sigma_k_k_1 @ self.P_0_vers_1(theta)
            )

            ### Creating a reduced state vector and covariance matrix
            z_1_red_k_k_1 = np.array([z_1_k_k_1[0], z_1_k_k_1[3]])  # x1  & x1_dot
            sigma_1_red_k_k_1 = np.array(
                [
                    [sigma_1_k_k_1[0, 0], sigma_1_k_k_1[0, 3]],
                    [sigma_1_k_k_1[3, 0], sigma_1_k_k_1[3, 3]],
                ]
            )

            g_k = self.g_nc(z_1_red_k_k_1)

            ### The filtering step can be computed for the reduced system
            L_red_k_k = (
                sigma_1_red_k_k_1
                @ C_k.T
                @ np.linalg.inv(C_k @ sigma_1_red_k_k_1 @ C_k.T + V)
            )
            sigma_1_red_k_k = sigma_1_red_k_k_1 - L_red_k_k @ C_k @ sigma_1_red_k_k_1
            z_1_red_k_k = z_1_red_k_k_1 + L_red_k_k @ (s_k - g_k)

            ### Putting back the reduced vector into the main one as well as the covariance matrix
            z_1_k_k = z_1_k_k_1
            z_1_k_k[0], z_1_k_k[3] = z_1_red_k_k[0], z_1_red_k_k[1]

            sigma_1_k_k = sigma_1_k_k_1
            sigma_1_k_k[0, 0], sigma_1_k_k[0, 3] = (
                sigma_1_red_k_k[0, 0],
                sigma_1_red_k_k[0, 1],
            )
            sigma_1_k_k[3, 0], sigma_1_k_k[3, 3] = (
                sigma_1_red_k_k[1, 0],
                sigma_1_red_k_k[1, 1],
            )

            ### Going back to the original frame
            z_k_k = self.P_0_vers_1(theta) @ z_1_k_k
            sigma_k_k = self.P_0_vers_1(theta) @ sigma_1_k_k @ self.P_1_vers_0(theta)

        return z_k_k, sigma_k_k

    def prediction_step(self, z_k_k, sigma_k_k):
        z_k_1_k = self.A @ z_k_k
        sigma_k_1_k = self.A @ sigma_k_k @ self.A.T + self.W
        return z_k_1_k, sigma_k_1_k

    async def lock_node(self):
        await self.node.lock()

    def wait_for_variables(self, variables):
        aw(self.node.wait_for_variables(variables))
        aw(self.client.sleep(0.1))

    def set_multiple_variables(self, variables: dict):
        print(variables)
        aw(self.node.set_variables(variables))
        aw(self.client.sleep(0.1))

    def get_multiple_variables(self, variables: list) -> dict:
        self.wait_for_variables(variables)
        return {variable: self.node.v[variable] for variable in variables}

    async def sleep(self, duration):
        await self.client.sleep(duration)

    def getProxH(self):
        return list(self.node.v.prox.horizontal)

    def getSpeedR(self):
        return self.node.v.motor.right.speed * self.speedConversion

    def getSpeedL(self):
        return self.node.v.motor.left.speed * self.speedConversion

    def get_vertices_waypoint(self, xb, yb):
        vertices = np.array(
            [
                [xb + self.l, yb + self.l],
                [xb - self.l, yb + self.l],
                [xb - self.l, yb - self.l],
                [xb + self.l, yb - self.l],
            ]
        )
        return vertices

    def get_cone_angles_waypoint(self, pos_estimate, xb, yb):
        """_summary_
        This function compute the range within which the robot should point before moving to the waypoint
        with a margin
        Args:
            pos_estimate (1D np array with 2 variables): x,y position of the robot in mm
            xb (float): x position of the waypoint
            yb (float): y position of the waypoint
            margin (float) : number between 0 and 1. (1-2*margin) corresponds to the coverage of the angle
            theta_max - theta_min
        """
        vertices = self.get_vertices_waypoint(xb, yb)
        angles = []
        for vertex in vertices:
            delta = vertex - pos_estimate
            angles.append(np.arctan2(delta[1], delta[0]))
        angles = np.array(angles)
        theta_max = np.max(angles)
        theta_min = np.min(angles)
        delta = theta_max - theta_min
        return theta_max - self.coneMargin * delta, theta_min + self.coneMargin * delta

    def robot_align_waypoint(self, theta_estimate, theta_max, theta_min):
        if theta_estimate < theta_max and theta_estimate > theta_min:
            return True
        else:
            return False

    def translation_control(self, pos_estimate, xb, yb):
        x_estimate = pos_estimate[0]  # mm
        y_estimate = pos_estimate[1]  # mm
        theta_estimate = pos_estimate[2]  # rad

        x1_b = np.cos(theta_estimate) * xb + np.sin(theta_estimate) * yb  # mm
        x1_estimate = (
            np.cos(theta_estimate) * x_estimate + np.sin(theta_estimate) * y_estimate
        )  # mm

        u = self.K_translation * (x1_b - x1_estimate)  # en mm/s
        u = u / self.speedConversion
        if u > 225:
            u = 225
        left_motor_target = u  # en unité Thymio
        right_motor_target = u  # en unité Thymio
        return left_motor_target, right_motor_target

    def rotation_control(self, theta_estimate, xb, yb):
        theta_b = np.arctan2(yb, xb)  # en rad
        u = self.K_rotation * (theta_b - theta_estimate)  # en mm/s
        u = u / self.speedConversion
        if u > 225:
            u = 225
        left_motor_target = u  # en unité Thymio
        right_motor_target = -u  # en unité Thymio
        return left_motor_target, right_motor_target

    def navigate(self, current_pos, next_pos):
        pos_estimate = current_pos
        xb, yb = next_pos
        theta_max, theta_min = self.get_cone_angles_waypoint(pos_estimate[:2], xb, yb)
        if self.robot_align_waypoint(current_pos[-1], theta_max, theta_min):
            left, right = self.translation_control(pos_estimate, xb, yb)
        else:
            left, right = self.rotation_control(current_pos[-1], xb, yb)
        self.set_multiple_variables(
            {"motor.left.target": [int(left)], "motor.right.target": [int(right)]}
        )

    def robot_close_waypoint(self, pos_estimate, xb, yb):
        """_summary_
        Return a boolean to say if the robot is in a square with (xb,yb) as center and 2*l as length size
        Args:
            pos_estimate (1D np array with 2 variables): x,y position of the robot in mm
            xb (float): x position of the waypoint
            yb (float): y position of the waypoint
        """
        ones = np.array([1, 1, 1, 1])
        F = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        pos_waypoint = np.array([xb, yb])

        print("Is the goal reached ?")
        print(all(F @ (pos_estimate - pos_waypoint) <= self.l * ones))

        if all(F @ (pos_estimate - pos_waypoint) <= self.l * ones):
            return True
        else:
            return False

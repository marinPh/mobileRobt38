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
        self.speedConversion = 0.43478260869565216
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
        self.Ts = 1.1
        self.K_rotation = self.L / (self.Ts)
        self.K_translation = 1 / (self.Ts)

        self.W = np.diag([0.001, 0.001, 0.00001, 0.001, 0.001, 0.00001])
        self.V_c = np.diag([0.1, 0.1, 0.00001, 0.1, 0.00001])
        self.V_nc = np.diag([0.001, 0.00001])
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
        return None

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
        theta = z[2]
        s_nc = [
            z[3] * np.cos(theta)
            + z[4] * np.sin(theta),  # x_dot*cos(theta) + y_dot*sin(theta)
            self.L * z[5],
        ]  # L*theta_dot
        return np.array(s_nc)

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
        theta = z[2]
        x_dot, y_dot = z[3], z[4]
        grad = [
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

    def constructing_s(
        self,
        V_left_measure,
        V_right_measure,
        camera_working=False,
        x_measured=0,
        y_measured=0,
        theta_measured=0,
    ):
        s_nc = np.array(
            [
                (V_left_measure + V_right_measure) / 2,
                (V_left_measure - V_right_measure) / 2,
            ]
        )
        if camera_working:
            s_c = np.append(np.array([x_measured, y_measured, theta_measured]), s_nc)
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
                            and angle (float) in rad of the detected obstacle 0 is in front of thymio.
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
                pos.append((distance*100, angle))
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
        C_k = self.grad_g_c(z_k_k_1) if camera_working else self.grad_g_nc(z_k_k_1)
        V = self.V_c if camera_working else self.V_nc
        s_k = self.constructing_s(
            V_left_measure,
            V_right_measure,
            camera_working,
            x_measured,
            y_measured,
            theta_measured,
        )
        g_k = self.g_c(z_k_k_1) if camera_working else self.g_nc(z_k_k_1)

        ### The real filtering step that can be rewritten without any problem
        L_k_k = sigma_k_k_1 @ C_k.T @ np.linalg.inv(C_k @ sigma_k_k_1 @ C_k.T + V)
        sigma_k_k = sigma_k_k_1 - L_k_k @ C_k @ sigma_k_k_1
        z_k_k = z_k_k_1 + L_k_k @ (s_k - g_k)

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
        
    def set_multiple_variables(self, variables:dict):
        print(variables)
        aw(self.node.set_variables(variables))
        aw(self.client.sleep(0.1))
        
    def get_multiple_variables(self, variables:list) -> dict:
        self.wait_for_variables(variables)
        return {variable: self.node.v[variable] for variable in variables}

    async def sleep(self, duration):
        await self.client.sleep(duration)

    def getProxH(self):
        self.wait_for_variables(["prox.horizontal"])
        return list(self.node.v.prox.horizontal)
    
    def getSpeedR(self):
        self.wait_for_variables(["motor.right.speed"])
        return self.node.v.motor.right.speed * self.speedConversion

    def getSpeedL(self):
        self.wait_for_variables(["motor.left.speed"])
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

        u = self.K_translation * (x1_b - x1_estimate)
        left_motor_target = u  # en mm/s
        right_motor_target = u  # en mm/s
        return left_motor_target, right_motor_target

    def rotation_control(self, theta_estimate, xb, yb):
        theta_b = np.arctan2(yb, xb)  # en rad
        u = self.K_rotation * (theta_b - theta_estimate)
        left_motor_target = u  # en mm/s
        right_motor_target = -u  # en mm/s
        return left_motor_target, right_motor_target

    def navigate(self, current_pos, next_pos):
        pos_estimate = current_pos
        xb, yb = next_pos
        theta_max, theta_min = self.get_cone_angles_waypoint(pos_estimate[:2], xb, yb)
        if self.robot_align_waypoint(current_pos[-1], theta_max, theta_min):
            left, right = self.translation_control(pos_estimate, xb, yb)
        else:
            left, right = self.rotation_control(current_pos[-1], xb, yb)
        right, left = right / self.speedConversion, left / self.speedConversion
        self.set_multiple_variables({"motor.left.target": [int(left)/self.speedConversion], "motor.right.target": [int(right)/self.speedConversion]})

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

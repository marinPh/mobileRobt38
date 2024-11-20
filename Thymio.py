from tdmclient import ClientAsync, aw
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import cv2
import numpy as np
from heapq import heappush, heappop


class Thymio:
    async def __init__(self):
        self.client = ClientAsync()
        self.node = await self.client.wait_for_node()
        await self.node.lock()
        self.ratio =  5/(4003-1455)
        self.sensorAngles = {
    'left_front': -30,
    'front middle-left': -15,
    'front middle': 0,
    'front middle-right':15,
    'front right': 30,
    'left_back': -135,
    'right_back': 135,
}
        
        self.L = 1 
        self.Ts = 0.05
        self.K_rotation = self.L/self.Ts
        self.K_translation = 1/self.Ts
        
        
        
        self.W = np.identity(6)
        self.V_c = np.identity(4)
        self.V_nc = np.identity(2)
        self.A = np.array([[1, 0, 0, self.Ts, 0, 0],
              [0, 1, 0, 0, self.Ts, 0],
              [0, 0, 1, 0, 0, self.Ts],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
        return None
    
    def g_c(self,z):
        theta = z[2]
        s_c=[z[0], # x
            z[1], # y
            z[3]*np.cos(theta) + z[4]*np.sin(theta), # x_dot*cos(theta) + y_dot*sin(theta)
            self.L*z[5]] # L*theta_dot
        return np.array(s_c)

    def g_nc(self,z):
        theta = z[2]
        s_nc=[z[3]*np.cos(theta) + z[4]*np.sin(theta), # x_dot*cos(theta) + y_dot*sin(theta)
        self.L*z[5]] # L*theta_dot
        return np.array(s_nc)

    def grad_g_c(self,z):
        theta = z[2]
        x_dot, y_dot = z[3], z[4]
        grad = [[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -x_dot*np.sin(theta) + y_dot*np.cos(theta), np.cos(theta), np.sin(theta), 0],
                [0, 0, 0, 0, 0, self.L]]
        return np.array(grad)

    def grad_g_nc(self,z):
        theta = z[2]
        x_dot, y_dot = z[3], z[4]
        grad = [[0, 0, -x_dot*np.sin(theta) + y_dot*np.cos(theta), np.cos(theta), np.sin(theta), 0],
                [0, 0, 0, 0, 0, self.L]]
        return np.array(grad)
    
    def constructing_s(V_left_measure, V_right_measure, camera_working=False, x_measured=0, y_measured=0):
        s_nc = np.array([(V_left_measure + V_right_measure)/2,
                         (V_left_measure - V_right_measure)/2])
        if camera_working:
            s_c = np.append(np.array([x_measured,y_measured]),
                            s_nc)
            return s_c
        return s_nc
    
    
    def getObstaclePosition(self) -> list:
        """
    Calculates the positions of obstacles based on sensor readings.
    The function iterates through 7 sensors, calculates the distance and angle
    for each sensor, and appends the position to a list. If the distance is greater
    than 5, it appends (-1, 0) to indicate no obstacle detected within the threshold.
    Returns:
        list of tuples: A list of tuples where each tuple contains the distance (float)
                        and angle (float) of the detected obstacle.
    """
        pos = []
        detected = False
        for i in range(7):
            ##if 2 sensors are close to each other link the 2 dots

            prox = self.getProxH()
            angle = self.sensorAngles[list(self.sensorAngles.keys())[i]]
            distance = self.ratio*(4003-list(prox)[i])
            if distance > 5:
                pos.append((-1,0))
            else:
                detected = True
                pos.append((distance,angle))
        return pos, detected
    
    def filtering_step(self,z_k_k_1, sigma_k_k_1, V_left_measure, V_right_measure, 
                       camera_working=False, x_measured=0, y_measured=0):
        ### Computing the variables that are dependant on the state of the camera
        C_k = self.grad_g_c(z_k_k_1) if camera_working else self.grad_g_nc(z_k_k_1)
        V = self.V_c if camera_working else self.V_nc
        s_k = self.constructing_s(V_left_measure, V_right_measure, camera_working, x_measured, y_measured)
        g_k = self.g_c(z_k_k_1) if camera_working else self.g_nc(z_k_k_1)

        ### The real filtering step that can be rewritten without any problem
        L_k_k = sigma_k_k_1@C_k.T@np.linalg.inv(C_k@sigma_k_k_1@C_k.T + V)
        sigma_k_k = sigma_k_k_1 - L_k_k@C_k@sigma_k_k_1
        z_k_k = z_k_k_1 + L_k_k@(s_k-g_k)

        return z_k_k, sigma_k_k
        
    def prediction_step(self,z_k_k, sigma_k_k):
        z_k_1_k = self.A@z_k_k
        sigma_k_1_k = self.A@sigma_k_k@self.A.T + self.W
        return z_k_1_k, sigma_k_1_k


    async def lock_node(self):
        await self.node.lock()

    def wait_for_variables(self, variables):
        aw(self.node.wait_for_variables(variables))

    async def sleep(self, duration):
        await self.client.sleep(duration)
        
    async def set_var(self, var, value):
        await self.node.set_var(var, value)
        
    async def getProxH(self):
        self.wait_for_variables(["prox.horizontal"])
        aw(self.client.sleep(0.1))
        return self.node.v.prox.horizontal
    
    async def getWheelR(self):
        self.wait_for_variables(["motor.right.speed"])
        aw(self.client.sleep(0.1))
        return self.node.v.motor.right.speed
    
    async def getWheelL(self):
        self.wait_for_variables(["motor.left.speed"])
        aw(self.client.sleep(0.1))
        return self.node.v.motor.left.speed
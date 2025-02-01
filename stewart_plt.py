import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

class StewartPlatform:
    def __init__(self, r_B, r_P, lhl, ldl, gamma_B, gamma_P, ref_rotation=0):
        """
        Initialize the Stewart Platform parameters
        :param r_B: Radius of the base
        :param r_P: Radius of the platform
        :param lhl: Length of the upper leg
        :param ldl: Length of the lower leg
        :param gamma_B: Offset angle of base
        :param gamma_P: Offset angle of platform
        :param ref_rotation: Initial rotation of the platform (optional)
        """
        self.r_B, self.r_P = r_B, r_P
        self.lhl, self.ldl = lhl, ldl
        self.gamma_B, self.gamma_P = gamma_B, gamma_P
        self.ref_rotation = ref_rotation
        self.init_geometry()

    def init_geometry(self):
        """
        Initialize the geometry of the Stewart Platform.
        This sets up the positions of the base (B) and platform (P) attachment points.
        """
        pi = np.pi
        beta = np.array([pi/2 + pi, pi/2, 2*pi/3 + pi/2 + pi, 2*pi/3 + pi/2, 4*pi/3 + pi/2 + pi, 4*pi/3 + pi/2])
        psi_B = np.array([-self.gamma_B, self.gamma_B, 2*pi/3 - self.gamma_B, 2*pi/3 + self.gamma_B, 4*pi/3 - self.gamma_B, 4*pi/3 + self.gamma_B])
        psi_P = np.array([pi/3 + 4*pi/3 + self.gamma_P, pi/3 - self.gamma_P, pi/3 + self.gamma_P, pi/3 + 2*pi/3 - self.gamma_P, pi/3 + 2*pi/3 + self.gamma_P, pi/3 + 4*pi/3 - self.gamma_P])
        
        # Base and platform points (B and P) in 3D space
        self.B = self.r_B * np.array([[np.cos(psi_B[i]), np.sin(psi_B[i]), 0] for i in range(6)]).T
        self.P = self.r_P * np.array([[np.cos(psi_P[i]), np.sin(psi_P[i]), 0] for i in range(6)]).T
        
        # Home position for the platform, assuming Z is adjustable
        z_home = np.sqrt(self.ldl**2 - np.linalg.norm(self.P[:2] - self.B[:2], axis=0)**2)
        self.home_pos = np.array([0, 0, np.mean(z_home)])
        self.angles = np.zeros(6)

    def rot_matrix(self, angles):
        """
        Generate the rotation matrix based on given Euler angles (x, y, z)
        :param angles: Rotation angles along X, Y, Z axes
        :return: Rotation matrix
        """
        Rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def calculate_angles(self, trans, rot):
        """
        Compute the leg angles based on the translation (trans) and rotation (rot)
        using inverse kinematics.
        :param trans: Translation vector [x, y, z]
        :param rot: Rotation vector [rx, ry, rz]
        :return: Updated leg angles
        """
        R = self.rot_matrix(rot)
        
        # Calculate the leg vectors based on translation and rotation
        leg_vectors = trans[:, np.newaxis] + self.home_pos[:, np.newaxis] + R @ self.P - self.B
        leg_lengths = np.linalg.norm(leg_vectors, axis=0)
        
        # Calculating angles using the leg lengths
        g = leg_lengths**2 - (self.ldl**2 - self.lhl**2)
        e = 2 * self.lhl * leg_vectors[2, :]
        f = 2 * self.lhl * (np.cos(self.angles) * leg_vectors[0, :] + np.sin(self.angles) * leg_vectors[1, :])
        
        # Calculate the angles for each leg
        self.angles = np.arcsin(g / np.sqrt(e**2 + f**2)) - np.arctan2(f, e)
        return self.angles

    def plot_platform(self, ax, trans, rot):
        """
        Plot the Stewart Platform with updated positions based on translation and rotation.
        :param ax: Matplotlib 3D axis object
        :param trans: Translation vector [x, y, z]
        :param rot: Rotation vector [rx, ry, rz]
        """
        R = self.rot_matrix(rot)
        P_trans = trans[:, np.newaxis] + R @ self.P
        
        ax.clear()
        
        # Plot the platform (hexagonal shape)
        ax.add_collection3d(Poly3DCollection([P_trans.T], facecolors='blue', alpha=0.3))
        
        # Plot the legs connecting the base to the platform
        for i in range(6):
            ax.plot([self.B[0, i], P_trans[0, i]], 
                    [self.B[1, i], P_trans[1, i]], 
                    [self.B[2, i], P_trans[2, i]], 'black')

        # Set the limits and labels for the plot
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 200)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

# NOTE: The right-handed coordinate frame is as follows:
#       X -- FORWARD
#       Y -- RIGHT
#       Z -- UPWARD
# NOTE: Joint angles correspond to the following movements
#       0 -- SHOULDER ADDUCTION / ABDUCTION (+/- Y)
#       1 -- SHOULDER EXTENSION / FLEXION (+/- X)
#       2 -- HUMERAL MEDIAL/LATERAL ROTATION (+/- Z)
#       3 -- ELBOW FLEXION / EXTENSION (+/- X)
#       4 -- WRIST PRONATION / SUPINATION (+/- Z)
#       5 -- WRIST RADIAL / ULNAR DEVIATION (+/- X)
#       6 -- WRIST FLEXION / EXTENSION (+/- Y)

import numpy as np

from ...utils import Quaternion as quat
from .. import AbstractBaseFilter

class UpperLimbDHFilter(AbstractBaseFilter):
    """ A Python implementation of a 7-DOF upper limb Denavit-Hartenberg robotic model """
    N_JOINTS = 8
    GRAVITY = 9.80665                   # gravitational constant
    TAL = [ [ 0.285, 0.269, 0.158 ], 
            [ 0.276, 0.265, 0.121 ], 
            [ 0.288, 0.235, 0.184 ] ]   # rotation constants (upper arm, forearm, hand)
                                        # transverse, anteroposterior, lateral
    ZO = np.array( [ 0, 0, 1 ] )

    # NOTE: height (m) and weight (kg)
    def __init__(self, height = 1.80, weight = 80):
        """ 
        Constructor

        Parameters
        ----------
        height : float
            The height of the individual
        weight : float
            The weight of the individual

        Returns
        -------
        obj
            An UpperLimbDHFilter object
        """
        # Model parameters
        self.length = [ i * height for i in [ 0.1111, 0.174, 0.156, 0.1079 ] ]  # (sternum --> shoulder), (shoulder --> elbow), (elbow --> wrist), (wrist --> palm)
        # self.length = [ i * height for i in [ 0.174, 0.156, 0.1079 ] ]  # (shoulder --> elbow), (elbow --> wrist), (wrist --> palm)
        self.mass = [ i * weight for i in [ 0.0271, 0.0162, 0.0061 ] ]  # upper arm, forearm, hand
        
        # Biological parameters
        self.__mass = None
        self.__com = None
        self.__moi = None
        self.__compute_bio_params()

        # Denavit-Hartenberg parameters
        self.__o = None
        self.__d = None
        self.__r = None 
        self.__a = None
        self.__compute_dh_params()

    def __compute_bio_params(self):
        """
        Computes the biological parameters needed for the model including: center of mass and moment of inertia

        Notes
        -----
        Center of mass (CoM) and Moment of Inertia (MoI) formulations are taken from
            Determining Upper Limb Kinematics and Dynamics During Everyday Tasks
            Ingarm A. Murray, B.Sc. (Hons.), M.Sc.
            Centre for Rehabilitation and Engineering Studies
            Department of Mechanical, Materials and Manufacturing Engineering
            University of Newcastle upon Tyne
            November 1999 
        """
        k = UpperLimbDHFilter.TAL                                                       # rotation constants
        com = [ i * j for i,j in zip( [ 0.5772, 0.4574, 0.3624 ], self.length[1:] ) ]   # upper arm, forearm, hand
        self.__mass = [ 0, 0, self.mass[0], 0, self.mass[1], 0, 0, self.mass[2] ]       # DH link mass
        self.__com = [ np.array( [ 0, 0, 0 ] ),
                       np.array( [ 0, 0, 0 ] ),
                       np.array( [ 0, 0, -com[0] ] ),
                       np.array( [ 0, 0, 0 ] ),
                       np.array( [ 0, 0, -com[1] ] ),
                       np.array( [ 0, 0, 0 ] ),
                       np.array( [ 0, 0, 0 ] ),
                       np.array( [ com[2], 0, 0 ] ) ]                                   # DH link CoM
        self.__moi = [ np.array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] ),
                       np.array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] ),
                       np.array( [ [ self.mass[0] * ( k[0][1] * com[0] ) ** 2, 0, 0 ], 
                                   [ 0, self.mass[0] * ( k[0][0] * com[0] ) ** 2, 0 ],
                                   [ 0, 0, self.mass[0] * ( k[0][2] * com[0] ) ** 2 ] ] ),
                       np.array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] ),
                       np.array( [ [ self.mass[1] * ( k[1][0] * com[1] ) ** 2, 0, 0 ], 
                                   [ 0, self.mass[1] * ( k[1][1] * com[1] ) ** 2, 0 ],
                                   [ 0, 0, self.mass[1] * ( k[1][2] * com[1] ) ** 2 ] ] ),
                       np.array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] ),
                       np.array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] ),
                       np.array( [ [ 0, 0, self.mass[2] * ( k[2][2] * com[2] ) ** 2 ], 
                                   [ 0, self.mass[2] * ( k[2][0] * com[2] ) ** 2, 0 ],
                                   [ 0, 0, self.mass[2] * ( k[2][1] * com[2] ) ** 2 ] ] ) ]       # DH link MoI [ Ixx, Iyy, Izz ]

    def __compute_dh_params(self):
        """
        Compute the Denavit-Hartenberg parameters for the robotic model

        Notes
        -----
        Although this function returns nothing, the following internal variables are set:
        __o (theta offset), 
        __a (angle about common normal, from old z axis to new z axis), 
        __d (offset along previous z to common normal), 
        __r (length of the common normal)
        """
        self.__o = np.deg2rad( [ -90, 90, 90,   0,   0,  0, 90,  0 ] ).tolist()
        self.__a = np.deg2rad( [  90, 90, 90,  90, -90, 90, 90, 90 ] ).tolist()    
        self.__d = [ 0, 0, 0, -self.length[1], 0, -self.length[2], 0, 0 ]
        self.__r = [ -self.length[0], 0, 0, 0, 0, 0, 0, -self.length[3] ]

    def __dh_frame_transformation(self, src, dest, q):
        """
        Computes the frame transformation from one link to the next given the DH parameters

        Parameters
        ----------
        src : int
            The initial link
        dest : int
            The terminal link
        q : numpy.ndarray (n_joints,)
            The current angles for each joint

        Returns
        -------
        numpy.ndarray (4,4)
            The homogenous transformation matrix from src to dest
        """
        T = np.eye( 4 )
        for i in range( src+1, dest+1 ):
            cq = np.cos( q + self.__o[i] )
            sq = np.sin( q + self.__o[i] )
            ca = np.cos( self.__a[i] )
            sa = np.sin( self.__a[i] )

            Ti = np.array( [ [ cq, -sq * ca,  sq * sa, self.__r[i] * cq ], 
                             [ sq,  cq * ca, -cq * sa, self.__r[i] * sq ], 
                             [ 0,        sa,       ca,      self.__d[i] ], 
                             [ 0,         0,        0,                1 ] ] )
            T = T @ Ti
        return T

    def __forward_kinematics(self, q):
        """
        Computes the forward kinematic transformations for each joint from the base

        Parameters
        ----------
        q : numpy.ndarray (n_joints,)
            The current angles for each joint
        
        Returns
        -------
        iterable of numpy.ndarray (4, 4)
            A list of homogenous transformation matrices for the shoulder, elbow, wrist, and hand
        """
        q = np.hstack( [ 0, q ] )
        T = [ np.eye( 4 ) ]
        for i in range( UpperLimbDHFilter.N_JOINTS ):
            Ti = np.array( [ [ np.cos( q[i] + self.__o[i] ), -np.sin( q[i] + self.__o[i] ) * np.cos( self.__a[i] ),  np.sin( q[i] + self.__o[i] ) * np.sin( self.__a[i] ), self.__r[i] * np.cos( q[i] + self.__o[i] ) ],
                             [ np.sin( q[i] + self.__o[i] ),  np.cos( q[i] + self.__o[i] ) * np.cos( self.__a[i] ), -np.cos( q[i] + self.__o[i] ) * np.sin( self.__a[i] ), self.__r[i] * np.sin( q[i] + self.__o[i] ) ],
                             [                            0,                                 np.sin( self.__a[i] ),                                 np.cos( self.__a[i] ),                                self.__d[i] ],
                             [                            0,                                                     0,                                                     0,                                          1 ] ] )
            T.append( T[-1] @ Ti )
        mask = [ 1, 4, 6, 8 ]   # shoulder, elbow, wrist, hand
        return [ T[idx] for idx in mask ]

    def __inverse_dynamics(self, q, qdot, qddot):
        """
        Computes the inverse dynamics for each joint of the robotic model using the recursive Euler-Newton method

        Parameters
        ----------
        q : numpy.ndarray (n_samples, n_joints)
            The current angles for each joint
        qdot : numpy.ndarray (n_samples, n_joints)
            The current velocity for each joint
        qddot : numpy.ndarray (n_samples, n_joints)
            The current acceleration for each joint
        
        Returns
        -------
        numpy.ndarray
            The torque required by each joint for these specified dynamics [n_samples x n_joints]
        """
        n_samples, n_dof = q.shape

        q = np.hstack( [ np.zeros( ( n_samples, 1 ) ), q ] )
        qdot = np.hstack( [ np.zeros( ( n_samples, 1 ) ), qdot ] )
        qddot = np.hstack( [ np.zeros( ( n_samples, 1 ) ), qddot ] )

        n_dof += 1

        grav = -1 * UpperLimbDHFilter.ZO * UpperLimbDHFilter.GRAVITY
        tau = np.zeros( ( n_samples, n_dof ) )

        for k in range( n_samples ):
            w, v = [], []
            wdot, vdot = [], []
            n, f = [], []

            # forward recursion
            for i in range( n_dof ):
                R = self.__dh_frame_transformation( i-1, i, q[k,i] )[:-1, :-1]
                p = np.array( [ self.__r[i], self.__d[i] * np.sin( self.__a[i] ), self.__d[i] * np.cos( self.__a[i] ) ] )

                if i == 0:
                    w.append( R.T @ ( UpperLimbDHFilter.ZO * qdot[k,i] ) )
                    wdot.append( R.T @ ( UpperLimbDHFilter.ZO * qddot[k,i] ) )
                    vdot.append( R.T @ grav + np.cross( wdot[-1], p ) + np.cross( w[-1], np.cross( w[-1], p ) ) )
                else:
                    w.append( R.T @ ( w[-1] + UpperLimbDHFilter.ZO * qdot[k,i] ) )
                    wdot.append( R.T @ ( wdot[-1] + UpperLimbDHFilter.ZO * qddot[k,i] 
                                       + np.cross( w[-2], UpperLimbDHFilter.ZO * qdot[k,i] ) ) )
                    vdot.append( R.T @ vdot[-1] + np.cross( wdot[-1], p ) + np.cross( w[-1], np.cross( w[-1], p ) ) )

            # backward recursion
            for i in reversed( range( n_dof ) ):
                p = np.array( [ self.__r[i], self.__d[i] * np.sin( self.__a[i] ), self.__d[i] * np.cos( self.__a[i] ) ] )
                vcdot = vdot[i] + np.cross( wdot[i], self.__com[i] ) + np.cross( w[i], np.cross( w[i], self.__com[i] ) )

                F = self.__mass[i] * vcdot
                N = self.__moi[i] @ wdot[i] + np.cross( w[i], self.__moi[i] @ w[i] )

                if i == ( n_dof - 1 ):
                    n.append( np.cross( self.__com[i] + p, F ) + N )
                    f.append( F )
                else:
                    R = self.__dh_frame_transformation( i, i+1, q[k,i+1] )[:-1, :-1]
                    n.append( R @ ( n[-1] + np.cross( R.T @ p, f[-1] ) ) + np.cross( self.__com[i] + p, F ) + N )
                    f.append( R @ f[-1] + F )

                R = self.__dh_frame_transformation( i-1, i, q[k,i] )[:-1, :-1]
                tau[k,i] = n[-1] @ R.T @ UpperLimbDHFilter.ZO
        return np.squeeze( tau[:,1:] )

    def set_mass(self, value, link):
        """
        Set the mass of a specified link in the model

        Parameters
        ----------
        value : float
            The mass of the link (in kg)
        link : str
            The link whose mass we setting {'upper_arm', 'forearm', 'hand'}
        """
        links = [ 'upper_arm', 'forearm', 'hand' ]
        self.mass[ links.index( link ) ] = value
        self.__compute_bio_params()

    def set_length(self, length, link):
        """
        Set the length of a specified link in the model

        Parameters
        ----------
        length : float
            The length of the link (in meters)
        link : str
            The link whose length we are setting {'chest', 'upper_arm', 'forearm', 'hand'}
        """
        links = [ 'chest', 'upper_arm', 'forearm', 'hand' ]
        self.length[ links.index( link ) ] = value
        self.__compute_bio_params()
        self.__compute_dh_params()

    def filter(self, q, qdot, qddot):
        """
        Process input joint position, velocity, and acceleration into model kinematics and dynamics

        Parameters
        ----------
        q : numpy.ndarray (n_samples, n_joints)
            The current angles for each joint
        qdot : numpy.ndarray (n_samples, n_joints)
            The current velocity for each joint
        qddot : numpy.ndarray (n_samples, n_joints)
            The current acceleration for each joint

        Returns
        -------
        numpy.ndarray (n_samples, 12)
            The position of the shoulder, elbow, wrist, and hand
        numpy.ndarray (n_samples, n_joints)
            The torque required by each joint for these specified dynamics
        """
        if len( q.shape ) == 1: q = np.expand_dims( q, axis = 0 )
        if len( qdot.shape ) == 1: qdot = np.expand_dims( qdot, axis = 0 )
        if len( qddot.shape ) == 1: qddot = np.expand_dims( qddot, axis = 0 )

        n_samples = q.shape[0]
        positions = np.zeros( ( n_samples, 3 * 4 ) )

        # kinematic analysis 
        for i in range( n_samples):  
            T = self.__forward_kinematics( q[i,:] )
            for j in range( 4 ): 
                positions[i,(3*j):(3*(j+1))] = T[j][:-1,-1]

        # dynamic analysis
        torques = self.__inverse_dynamics( q, qdot, qddot )
        
        return positions, torques

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    N_SAMPLES = 100
    N_JOINTS = 7

    timestamps = np.linspace( 0, 5, N_SAMPLES )
    angles = np.pi * np.sin( np.linspace( 0, 6 * np.pi, N_SAMPLES ) )
    velocities = np.gradient( angles, timestamps )
    accelerations = np.gradient( velocities, timestamps )

    filt = UpperLimbDHFilter()

    fig = plt.figure()
    plot_color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i in range( N_JOINTS ):
        all_angles = np.zeros( ( N_SAMPLES, N_JOINTS ) )
        all_velocities = np.zeros( ( N_SAMPLES, N_JOINTS ) )
        all_accelerations = np.zeros( ( N_SAMPLES, N_JOINTS ) )

        all_angles[:, i] = angles
        all_velocities[:, i] = velocities
        all_accelerations[:, i] = accelerations

        t0 = time.perf_counter()
        pos, trq = filt.filter( all_angles, all_velocities, all_accelerations )
        tf = time.perf_counter()
        print( 'AVG PROCESSING TIME: % .2f ms' % ( 1e3 * ( tf - t0 ) / N_SAMPLES ) )

        joint_moves = [ 'SHLDR\nAB/AD', 'SHLDR\nFLEX/EXT', 'HUM\nROT', 'ELBOW\nFLEX/EXT', 
                        'WRIST\nPRO/SUP', 'WRIST\nDEV', 'WRIST\nFLEX/EXT' ]
        for j in range( N_JOINTS ):
            ax = fig.add_subplot( N_JOINTS, N_JOINTS, j+1 + N_JOINTS * i )
            ax.plot( timestamps, trq[:,j], color = plot_color[j] )
            ax.set_ylim( [-30, 30] )
            
            if i != ( N_JOINTS - 1 ): ax.set_xticks( [] )
            if j != 0: ax.set_yticks( [] )

            if i == 0: ax.set_title( 'Joint %d' % (j+1), fontweight = 'bold' )
            if j == 0: ax.set_ylabel( joint_moves[i], fontweight = 'bold' )

    plt.show()
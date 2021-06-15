import numpy as np

from ...utils import Quaternion as quat
from .. import AbstractBaseFilter

class QuaternionJointAngleFilter(AbstractBaseFilter):
    """ A Python implementation of a quaternion joint angle filter """
    STANDARD_BASIS = [ np.array( [ 1, 0, 0 ] ), 
                       np.array( [ 0, 1, 0 ] ),
                       np.array( [ 0, 0, 1 ] ) ]

    @staticmethod
    def __compute_hinge_angle( q1, q2, vref ):
        """
        Computes the angle of a hinge joint given two quaternion measurements

        Parameters
        ----------
        q1 : numpy.ndarray
            Quaternion orientation of the first device
        q2 : numpy.ndarray
            Quaternion orientation of the second device
        vref : numpy.ndarray
            Reference vector to determine angle sign
        
        Returns
        -------
        float
            Angle of the hinge joint
        """
        qr = quat.relative( q1, q2 )
        vref = quat.rotate( q1, vref )
        angle, axis = quat.to_axis_angle( qr )
        return angle * np.sign( np.dot( vref, axis ) )

    @staticmethod
    def __compute_spherical_angle( q1, q2 ):
        """
        Computes the angle of a hinge joint given two quaternion measurements

        Parameters
        ----------
        q1 : numpy.ndarray
            Quaternion orientation of the first device
        q2 : numpy.ndarray
            Quaternion orientation of the second device

        Returns
        -------
        numpy.ndarray
            The 3 angles of the spherical joint
        """
        qr = quat.relative( q1, q2 )
        angles = np.zeros( 3, dtype = np.float32 )

        for i in range( 3 ):
            e = QuaternionJointAngleFilter.STANDARD_BASIS[i]
            _, twist = quat.to_swing_twist( qr, e )
            angle, axis = quat.to_axis_angle( twist )
            angles[i] = angle * np.sign( np.dot( e, axis ) )
        return angles

    def __init__(self, joints = [ 'shoulder', 'elbow', 'wrist' ] ):
        """
        Constructor

        Parameters
        ----------
        joints : iterable of str
            The joints that we are measuring the angles of {'shoulder', 'elbow', 'wrist'}
        
        Returns
        -------
        obj
            A QuaternionJointAngleFilter object
        """
        self.__joints = joints

    def filter(self, q ):
        """
        Computes joint angles from quaternion measurements

        Parameters
        ----------
        q : numpy.ndarray (n_samples, n_quats x 4)
            The quaternion measurements

        Returns
        -------
        np.ndarray (n_samples, n_joints)
            The measured joint angles
        """
        if len( q.shape ) == 1: q = np.expand_dims( q, axis = 0 )
        n_samps, n_quats = q.shape

        assert( np.mod( n_quats, 4 ) == 0 )
        n_quats = n_quats // 4
        assert( n_quats == ( len( self.__joints ) + 1 ) )

        all_angles = []
        for i in range( n_samps ):
            sample_angles = []
            for j in range( n_quats - 1 ):
                q1 = q[i, (4*j):(4*(j+1))]
                q2 = q[i, (4*(j+1)):(4*(j+2))]

                if self.__joints[j] in [ 'shoulder', 'wrist' ]:
                    angle = QuaternionJointAngleFilter.__compute_spherical_angle( q1, q2 )
                elif self.__joints[j] in [ 'elbow' ]:
                    vref = quat.rotate( q1, QuaternionJointAngleFilter.STANDARD_BASIS[2] )
                    angle = QuaternionJointAngleFilter.__compute_hinge_angle( q1, q2, vref )

                sample_angles.append( angle )
            all_angles.append( np.hstack( sample_angles ) )
        
        return np.vstack( all_angles )

if __name__ == '__main__':
    N_SAMPLES = 100
    N_QUATS = 4

    quats = np.random.random( ( N_SAMPLES, 4 * N_QUATS ) )
    for i in range( N_SAMPLES ):
        for j in range( N_QUATS ):
            quats[i, (4*j):(4*(j+1))] = quat.normalize( quats[i, (4*j):(4*(j+1))] )
    
    qja = QuaternionJointAngleFilter( joints = [ 'shoulder', 'elbow', 'wrist' ] )
    angles = qja.filter( quats )

    print( angles )
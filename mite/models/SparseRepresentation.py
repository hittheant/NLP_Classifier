import sys
import numpy as np
import numpy.matlib

from numpy import linalg as la
from sklearn.preprocessing import normalize
from . import AbstractBaseModel

try: import sparsesolvers as ss
except ImportError: pass

class SparseRepresentation(AbstractBaseModel):
    """ Python implementation of sparse representation model """
    def __init__(self, X, y, tol = 0.1):
        """
        Constructor

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,)
            Training labels
        tol : float
            Tolerance for homotopy L1-solver

        Returns
        -------
        obj
            A SparseRepresentation model
        """
        self.__dictionary = None    # X
        self.__labels = None        # y
        self.__tolerance = tol

        self.__class_vector = np.unique( y )
        self.__num_classes = np.size( self.__class_vector )

        self.train( X, y )

    def train(self, X, y):
        """
        Train the model

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,)
            Training labels
        """
        self.__dictionary = normalize( X.T, norm = 'l2', axis = 0 )
        self.__labels = y
    
    def predict(self, X):
        """
        Estimate output from given input

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Testing data

        Returns
        -------
        numpy.ndarray (n_samples,)
            Estimated output
        """
        if 'sparsesolvers' in sys.modules:
            if len( X.shape ) == 1: X = np.expand_dims( X, axis=0 )
            X = normalize( X.T, norm = 'l2', axis = 0 )
            residuals = np.zeros( ( X.shape[1], self.__num_classes ), dtype = np.float )
            for sample in range( 0, X.shape[1] ):
                x = X[ :, sample ]
                # solve the L1 minimization problem
                s, _ = ss.Homotopy( self.__dictionary ).solve( x, tolerance = self.__tolerance )

                # reconstruction residuals
                for ind in range( 0, self.__num_classes ):
                    c = self.__class_vector[ ind ]
                    coef_c = s[ np.equal( self.__labels, c ) ]
                    Dc = self.__dictionary[ :, np.equal( self.__labels, c ) ]
                    residuals[ sample, ind ] = la.norm( x - Dc @ coef_c, ord = 2 )

            return self.__class_vector[ np.argmin( residuals, axis = 1 ) ]
            # probs = np.divide( 1.0 - residuals, np.sum( 1.0 - residuals, axis = 1 )[ :, None ] ) if prob else None # TODO: LOOK UP BETTER METRIC
            # return predicts, probs
        else:
            raise NotImplementedError('Sparse solver module does not exist for non-Linux OS!')

    # if 'sparsesolvers' in sys.modules:
    #     # Note: Expects shape of X to be --> n_samples x n_features
    #     def predict(self, X, prob = False):
    #         if len( X.shape ) == 1: X = np.expand_dims( X, axis=0 )
    #         X = normalize( X.T, norm = 'l2', axis = 0 )
    #         residuals = np.zeros( ( X.shape[1], self.__num_classes ), dtype = np.float )
    #         for sample in range( 0, X.shape[1] ):
    #             x = X[ :, sample ]
    #             # solve the L1 minimization problem
    #             s, _ = ss.Homotopy( self.__dictionary ).solve( x, tolerance = self.__tolerance )

    #             # reconstruction residuals
    #             for ind in range( 0, self.__num_classes ):
    #                 c = self.__class_vector[ ind ]
    #                 coef_c = s[ np.equal( self.__labels, c ) ]
    #                 Dc = self.__dictionary[ :, np.equal( self.__labels, c ) ]
    #                 residuals[ sample, ind ] = la.norm( x - Dc @ coef_c, ord = 2 )

    #         predicts = self.__class_vector[ np.argmin( residuals, axis = 1 ) ]
    #         probs = np.divide( 1.0 - residuals, np.sum( 1.0 - residuals, axis = 1 )[ :, None ] ) if prob else None # TODO: LOOK UP BETTER METRIC
    #         return predicts, probs
    # else:
    #     def predict(self, X, prob = False):
    #         raise NotImplementedError('Sparse solver module does not exist for non-Linux OS!')

if __name__ == '__main__':
    from sklearn.datasets import load_digits, load_boston
    from sklearn.model_selection import train_test_split
    from ..utils.Metrics import confusion_matrix

    import matplotlib.pyplot as plt

    fig = plt.figure( figsize = (10.0, 5.0) )
    plt_count = 0

    for task in [ 'classification', 'regression' ]:
        ax = fig.add_subplot( 1, 2, plt_count + 1 )
        plt_count += 1
        
        if task == 'classification': data = load_digits()
        elif task == 'regression': data = load_boston()
    
        Xtrain, Xtest, ytrain, ytest = train_test_split( data.data, data.target, test_size = 0.33 )
        if task == 'classification': 
            mdl = SparseRepresentation( Xtrain, ytrain )        
            yhat = mdl.predict( Xtest )

        if task == 'classification': cm = confusion_matrix( ytest, yhat, labels = data.target_names, ax = ax, show = False )
        elif task == 'regression':
            ax.text( 0.4, 0.45, 'N/A', fontsize = 30 )
            # ax.plot( ytest, label = 'Actual' )
            # ax.plot( yhat, label = 'Predicted' )

            # ax.set_xlabel( 'Sample' )
            # ax.set_ylabel( 'Output Value' )
            # leg = ax.legend( frameon = False )

        ax.set_title( 'Digits Dataset Classification' if task == 'classification' else 'Boston Housing Regression' )
    plt.tight_layout()
    plt.show()
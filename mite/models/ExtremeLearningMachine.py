import numpy as np
import numpy.matlib
from sklearn.preprocessing import normalize

from . import AbstractBaseModel

class ExtremeLearningMachine(AbstractBaseModel):
    """ Python implementation of sparse representation model """
    @staticmethod 
    def sigmoid( x ):
        """
        Compute the sigmoid transformation

        Parameters
        ----------
        x : numpy.ndarray (m, n, k, ...)
            The sigmoidal input
        
        Returns
        -------
        numpy.ndarray (m, n, k, ...)
            The transformed values
        """
        return np.divide( 1.0, 1.0 + np.exp( -x ) )

    def __init__(self, X, y, n_hidden_layers = 100, activation_function = 'sigmoid', opt_lambda = np.arange( -4, 4, 0.2 ), regressor = False, seed = None ):
        """
        Constructor

        Parameters
        ----------
        ---------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,)
            Training labels
        n_hidden_layers : int
            The number of hidden layers for the ELM
        activation_function : str { 'sigmoid', 'tanh', 'relu' }
            Activation function for each node of model
        regressor : bool
            Flag to determine model-type (True if regressor, False if classifier)

        Returns
        -------
        obj
            An ExtremeLearningMachine model
        """

        self.__weights = None
        self.__bias = None
        self.__beta = None

        self.__n_layers = n_hidden_layers
        self.__opt_lambda = opt_lambda

        if activation_function == 'sigmoid':
            self.__activation_function = ExtremeLearningMachine.sigmoid
        self.__regressor = regressor
        self.__seed = seed

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

        # train ELM
        if self.__seed is not None: np.random.seed( self.__seed )
        X = normalize( X.T, norm = 'l2', axis = 0 )

        n_classes = np.unique( y ).shape[0]
        n_features, n_samples  = X.shape

        self.__weights = np.random.uniform( 0, 1, ( self.__n_layers, n_features ) )
        self.__bias = np.random.uniform( 0, 1, ( self.__n_layers ) )

        H = self.__activation_function( self.__weights @ X + np.matlib.repmat( self.__bias, n_samples, 1 ).T ).T
        
        # one-hot encoding
        Y = np.zeros( ( n_samples, n_classes ) )
        for c in range( 0, n_classes ):
            Y[:, c] = ( y == c )

        U, S, _ = np.linalg.svd( H.T @ H )
        A = H @ U
        B = A.T @ Y

        if self.__opt_lambda.shape[0] > 1:
            # leave one out analysis for optimal lambda value
            LOO = np.inf * np.ones( ( self.__opt_lambda.shape[0], ) )
            for i in range( self.__opt_lambda.shape[0] ):
                tmp = np.multiply( A, np.matlib.repmat( np.divide( 1.0, S + self.__opt_lambda[i] ), n_samples, 1 ) )
                hat = np.sum( np.multiply( tmp, A ), axis = 1 )
                yhat = tmp @ B

                errdiff = np.divide( ( Y - yhat ), np.matlib.repmat( 1.0 - hat, n_classes, 1 ).T )
                frob = np.linalg.norm( errdiff, ord = 'fro' )
                LOO[i] = frob ** 2 / n_samples
            opt = self.__opt_lambda[ np.argmin( LOO ) ]
        else:
            opt = self.__opt_lambda

        # compute beta values
        self.__beta = ( np.multiply( U, np.matlib.repmat( np.divide( 1, S + opt ), self.__n_layers, 1 ) ) @ B ).T
    
    def predict(self, X):
        X = normalize( X.T, norm = 'l2', axis = 0 )
        _, n_samples = X.shape

        H = self.__activation_function( self.__weights @ X + np.matlib.repmat( self.__bias, n_samples, 1 ).T )
        yhat = self.__beta @ H
        if self.__regressor: yhat = np.argmax( yhat )
        else: yhat = np.argmax( yhat, axis = 0 )
        return yhat


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
            mdl = ExtremeLearningMachine( Xtrain, ytrain )        
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
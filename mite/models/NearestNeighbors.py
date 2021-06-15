import numpy as np
from . import AbstractBaseModel

class NearestNeighbors(AbstractBaseModel):
    """ Python implementation of k-nearest neighbors model """
    def __init__(self, X, y, n_neighbors = 5, degree = 2, regressor = False):
        """
        Constructor

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,) if classifier, (n_samples, n_outputs) if regressor
            Training labels
        n_neighbors : int
            Number of neighbors to consider
        degree : int
            Degree of the Minkowski distance metric (2 is Euclidean distance, 1 is Manhattan distance, etc.)
        regressor : bool
            Flag to determine model-type (True if regressor, False if classifier)

        Returns
        -------
        obj
            A NearestNeighbors model
        """
        if regressor:
            from sklearn.neighbors import KNeighborsRegressor
            self.__model = KNeighborsRegressor( n_neighbors = n_neighbors, p = degree )
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.__model = KNeighborsClassifier( n_neighbors = n_neighbors, p = degree )

        self.train( X, y )

    def train(self, X, y):
        """
        Train the model

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Training data
        y : numpy.ndarray (n_samples,) if classifier, (n_samples, n_outputs) if regressor
            Training labels
        """
        self.__model = self.__model.fit( X, y )

    def predict(self, X):
        """
        Estimate output from given input

        Parameters
        ----------
        X : numpy.ndarray (n_samples, n_features)
            Testing data

        Returns
        -------
        numpy.ndarray (n_samples,) if classifier, (n_samples, n_outputs) if regressor
            Estimated output
        """
        return self.__model.predict( X )

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
        mdl = NearestNeighbors( Xtrain, ytrain, regressor = ( task == 'regression' ) )        
        yhat = mdl.predict( Xtest )

        if task == 'classification': cm = confusion_matrix( ytest, yhat, labels = data.target_names, ax = ax, show = False )
        elif task == 'regression':
            ax.plot( ytest, label = 'Actual' )
            ax.plot( yhat, label = 'Predicted' )

            ax.set_xlabel( 'Sample' )
            ax.set_ylabel( 'Output Value' )
            leg = ax.legend( frameon = False )

        ax.set_title( 'Digits Dataset Classification' if task == 'classification' else 'Boston Housing Regression' )
    plt.tight_layout()
    plt.show()
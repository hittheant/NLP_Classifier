import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from scipy import stats

def confusion_matrix(ytest, yhat, labels = [], cmap = 'viridis', ax = None, show = True):
    """
    Computes (and displays) a confusion matrix given true and predicted classification labels

    Parameters
    ----------
    ytest : numpy.ndarray (n_samples,)
        The true labels
    yhat : numpy.ndarray (n_samples,)
        The predicted label
    labels : iterable
        The class labels
    cmap : str
        The colormap for the confusion matrix
    ax : axis or None
        A pre-instantiated axis to plot the confusion matrix on
    show : bool
        A flag determining whether we should plot the confusion matrix (True) or not (False)

    Returns
    -------
    numpy.ndarray
        The confusion matrix numerical values [n_classes x n_classes]
    axis
        The graphic axis that the confusion matrix is plotted on or None
    """
    def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    cm = sk_confusion_matrix( ytest, yhat )
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:    
        fig = plt.figure()
        ax = fig.add_subplot( 111 )

    try:
        plt.set_cmap( cmap )
    except ValueError: cmap = 'viridis'

    im = ax.imshow( cm, interpolation = 'nearest', vmin = 0.0, vmax = 1.0, cmap = cmap )
    add_colorbar( im )

    if len( labels ):
        tick_marks = np.arange( len( labels ) )
        plt.xticks( tick_marks, labels, rotation=0 )
        plt.yticks( tick_marks, labels )

    thresh = 0.5 # cm.max() / 2.
    colors = mpl.cm.get_cmap( cmap )
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        r,g,b,_ = colors(cm[i,j])
        br = np.sqrt( r*r*0.241 + g*g*0.691 + b*b*0.068 )
        plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment = "center",
                    verticalalignment = 'center',
                    color = "black" if br > thresh else "white")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.tight_layout()
    if show: plt.show( block = True )
    
    return cm, ax

def statistical_significance( data1, data2, method = 'anova' ):
    """
    Computes the statistical significance of two datasets being different

    Parameters
    ----------
    data1 : numpy.ndarray
        The first dataset to compare
    data2 : numpy.ndarray
        The second dataset to compare
    method : str
        The statistical test to run {'t_test', 't_test_paired', 'anova', 'mann_whitney', 'wilcoxon', 'kruskal', 'friedman'}
    
    Returns
    -------
    iterable of floats
        The statistic of the test being run [0] and the resultant p-value [1]
    """
    function_dictionary = { 't_test'        : stats.ttest_ind,
                            't_test_paired' : stats.ttest_rel,
                            'anova'         : stats.f_oneway,
                            'mann_whitney'  : stats.mannwhitneyu,
                            'wilcoxon'      : stats.wilcoxon,
                            'kruskal'       : stats.kruskal,
                            'friedman'      : stats.friedmanchisquare }
    try:
        f = function_dictionary[method]
    except KeyError as e:
        print( 'Invalid method. Please choose a valid statistical test:', list( function_dictionary.keys() ) )
        raise e
    return f( data1, data2 )
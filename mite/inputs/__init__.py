# define abstract base class
import abc
import multiprocessing as mp

class AbstractBaseInput(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        """ Returns the associated name for the input device """
        raise NotImplementedError("Users must define 'name' to use this base class")

    @property
    @abc.abstractmethod
    def state(self):
        """ Returns the current state of the input device """
        raise NotImplementedError("Users must define 'state' to use this base class")

    @property
    @abc.abstractmethod
    def speriod(self):
        """ Returns the sampling period of the input device """
        raise NotImplementedError("Users must define 'speriod' to use this base class")

    @property
    @abc.abstractmethod
    def channelcount(self):
        """ Returns the number of channels for the input device """
        raise NotImplementedError("Users must define 'channelcount' to use this base class")

    @abc.abstractmethod
    def run(self, display=False):
        """ Starts the acquisition process of the input device """
        raise NotImplementedError("Users must define 'run' to use this base class")

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError("Users must define 'stop' to use this base class")
   
    @abc.abstractmethod
    def view(self):
        raise NotImplementedError("Users must define 'view' to use this base class")

    @abc.abstractmethod
    def hide(self):
        raise NotImplementedError("Users must define 'hide' to use this base class")

# import all submodules
from os.path import dirname, basename, isfile
import glob

modules = glob.glob( dirname( __file__ ) + "/*.py" )
__all__ = [ basename( f )[:-3] for f in modules if isfile( f ) and not f.endswith( '__init__.py' ) ]
for m in __all__: exec( 'from .' + m + ' import ' + m )

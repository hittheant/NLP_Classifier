# define abstract base class
import abc
class AbstractBaseProtocol(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def start(self):
        raise NotImplementedError("Users must define 'run' to use this base class")
        

# import all submodules
from os.path import dirname, basename, isfile
import glob

modules = glob.glob( dirname( __file__ ) + "/*.py" )
__all__ = [ basename( f )[:-3] for f in modules if isfile( f ) and not f.endswith( '__init__.py' ) ]
for m in __all__: exec( 'from .' + m + ' import ' + m )


import abc

from rl.dfo.config import DFOModelConfig

from IPython.core.debugger import set_trace

class DFOModel( abc.ABC ) :
    r"""Base class for any model used as component of a DFO agent

    This base class defines the blueprint for all DFO models, implemented in
    their specific backends. The API is made such that we pass the layers
    defined in a dictionary, and the specific backend instantiates these
    appropriately with their own low-level resources.

    Args:
        name (str): unique name of this model
        config (DFOModelConfig): Model configuration parameters

    """
    def __init__( self, name, config, verbose = True ) :
        super( DFOModel, self ).__init__()

        self._name = name
        self._config = config
        self._inputShape = config.inputShape
        self._outputShape = config.outputShape
        self._layersDefs = config.layersDefs
        self._useDiscreteOutputs = config.useDiscreteOutputs

        # backbone of the model, usually layers defining the bulk of the model
        self._backbone = []

        self._buildModel()
        
        if verbose :
            self._printConfig()


    @abc.abstractmethod
    def _buildModel( self ) :
        r"""Constructs the internal model using the appropriate backend
        
        Derived models for specific backends should use the layers definitions
        in order to create the appropriate model using the right backend calls.

        """
        pass


    @abc.abstractmethod
    def initialize( self, args = {} ) :
        r"""Initializes the model, if required, for the specific backend used

        Derived models might need to initialize resources (session, devices)
        in order to use them properly in the given backend.

        Args:
            args (dict): resources for initialization passed as a dictionary

        """
        pass


    @abc.abstractmethod
    def seed( self, seed ) :
        r"""Passes a given seed to the model to seed its generator

        Derived models need to implement this method according to the seeding
        method used in the specific implementation used.

        Args:
            seed (int): seed provided for the generator

        """
        pass


    @abc.abstractmethod
    def eval( self, x ) :
        r"""Implements forward pass of the model
        
        This forward pass should be implemented by each model in the specific
        backend, and should return the appropriate output of the model for the
        given input x, which should have a batch dimension added as well

        Args:
            x (np.ndarray): input to the model, with a bath dimension added

        """
        pass


    @abc.abstractmethod
    def copy( self, other ) :
        r"""Copies the weights of another model into this model

        This method should be implemented with the appropriate backend
        functionality, allowing to pass the weights of one network to another.

        Args:
            other (DFOModel): model from whom to copy the weights

        """
        pass


    @abc.abstractmethod
    def clone( self, mame = None ) :
        r"""Creates a new model, which is a copy of this model

        This method should be implemented with the appropriate backend
        functionality, allowing to pass the weights of one network to another.

        Args:
            name (str): name of this new cloned model

        """
        pass


    @abc.abstractmethod
    def weights( self ) :
        r"""Returns the weights of the model

        This method should be implemented with the appropriate backend
        functionality, allowing to grab the weights of the model as a 
        list of numpy ndarrays.

        Returns:
            (list): a list with the weights of each component as an element

        """
        pass


    @abc.abstractmethod
    def areWeightsEqual( self, otherWeights ) :
        r"""Compare this model's weights to another set of weights

        This method should be implemented with the appropriate backend
        functionality, allowing to compare if the weights given are
        the same as the weights of this model.

        Args:
            otherWeights (list): a list of np.ndarrays to compare to

        """
        pass


    @abc.abstractmethod
    def perturb( self, ptype, args ) :
        r"""Applies a small perturbation of a given type and given parameters
        
        Args:
            ptype (str): type of perturbation to be applied (uniform|gaussian)
            args (dict): parameters defining the perturbation to be applied
        Returns:
            (np.ndarray,np.ndarray): state of the random generator (bef.,now)

        """
        return (None,None)


    @abc.abstractmethod
    def save( self, filename ) :
        r"""Saves the model into a file for later usage

        Args:
            filename (str): file name (or fullpath) where to save the weights

        """
        pass


    @abc.abstractmethod
    def load( self, filename ) :
        r"""load a model from a given file

        Args:
            filename (str): file name (or fullpath) from whom to load the model

        """
        pass        


    @property
    def name( self ) :
        return self._name
    

    @property
    def inputShape( self ) :
        return self._inputShape
    

    @property
    def outputShape( self ) :
        return self._outputShape


    def _printConfig( self ) :
        # Each model could potentially override this with its own extra details
        print( '#############################################################' )
        print( '#                                                           #' )
        print( '#                 Model configuration                       #' )
        print( '#                                                           #' )
        print( '#############################################################' )

        print( 'model name          : ', self._name )
        print( 'input shape         : ', self._inputShape )
        print( 'output shape        : ', self._outputShape )
        print( 'discrete outputs    : ', str( self._useDiscreteOutputs ) )

        print( '#############################################################' )
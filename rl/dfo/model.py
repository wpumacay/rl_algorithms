
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
    def __init__( self, name, config ) :
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
        self._printConfig()


    @abc.abstractmethod
    def _buildModel( self ) :
        r"""Constructs the internal model using the appropriate backend
        
        Derived models for specific backends should use the layers definitions
        in order to create the appropriate model using the right backend calls.

        """
        pass


    @abc.abstractmethod
    def initialize( self, args ) :
        r"""Initializes the model, if required, for the specific backend used

        Derived models might need to initialize resources (session, devices)
        in order to use them properly in the given backend.

        Args:
            args (dict): resources for initialization passed as a dictionary

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
    def clone( self, other ) :
        r"""Clones the weights of one model into this model

        This method should also be implemented with the appropriate backend
        functionality, allowing to pass the weights of one network to another.

        Args:
            other (DFOModel): model from whom to copy the weights

        """
        pass


    @abc.abstractmethod
    def perturb( self, ptype, args ) :
        r"""Applies a small perturbation of a given type and given parameters
        
        Args:
            ptype (str): type of perturbation to be applied (uniform|gaussian)
            args (dict): parameters defining the perturbation to be applied

        """
        pass


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
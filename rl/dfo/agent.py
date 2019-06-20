
import abc
import numpy as np
from rl.dfo.model import DFOModel

class DFOAgent( abc.ABC ) :
    r"""Base class for all Derivate-Free-Optimization based agents
    
    Args:
        config (dict): configuration parameters for the agent
        model (DFOModel): model used by the agent

    """
    def __init__( self, name, config, model ) :
        super( DFOAgent, self ).__init__()

        self._name = name
        self._config = config
        self._model = model


    @abc.abstractmethod
    def act( self, state ) :
        r"""Returns an action by quering the model at the given state
            
        Args:
            state (np.ndarray): state representation to be used by the model

        """
        pass


    @abc.abstractmethod
    def onEndEpisode( self, args ) :
        r"""Updates agent internals when an episode ends

        Args:
            args (dict): parameters used to update internals of the agent

        """
        pass


    def clone( self, other ) :
        r"""Copies the whole model weights from another given model
        
        Args:
            other (DFOModel): model from which to copy the weights

        """
        if not self._model :
            print( 'ERROR> this agent has no model' )
        else :
            self._model.clone( other._model )


    def perturb( self, ptype, args ) :
        r"""Applies a small perturbation inside a gaussian distribution
        
        Args:
            ptype (str): type of perturbation to be applied (uniform|gaussian)
            args (dict): parameters defining the perturbation to be applied

        """
        if not self._model :
            print( 'ERROR> this agent has no model' )
        else :
            self._model.perturb( ptype, args )


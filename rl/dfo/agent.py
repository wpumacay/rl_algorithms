
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
        self._seed = 0


    def act( self, state ) :
        r"""Returns an action by quering the model at the given state
            
        Args:
            state (np.ndarray): state representation to be used by the model

        """
        if self._config.actionSpaceType == 'discrete' :
            # model should return probabilities of each action
            _actProbs = self._model.eval( state[np.newaxis,...] )
            if self._config.useDeterministicPolicy :
                return np.argmax( _actProbs )
            else :
                return np.random.choice( self._config.nActions, p = _actProbs )
        else :
            # the output comes from a gaussian, with mean given by model output
            return self._model.eval( state )

    @abc.abstractmethod
    def onStartEpisode( self, args = {} ) :
        r"""Configures agent internals when an episode is about to start

        Args:
            args (dict): parameters used to update internals of the agent

        """
        pass

    @abc.abstractmethod
    def update( self, transition ) :
        r"""Updates agent internals when a step is made

        Args:
            transition (tuple): a transition tuple ( s, a, r, s', done )

        """
        pass


    @abc.abstractmethod
    def onEndEpisode( self, args = {} ) :
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


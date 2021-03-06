
import abc
import numpy as np
from rl.pg.model import PGModel

class PGAgent( abc.ABC ) :
    r"""Base class for all Derivate-Free-Optimization based agents
    
    Args:
        config (dict): configuration parameters for the agent
        model (PGModel): model used by the agent

    """
    def __init__( self, name, config, model ) :
        super( PGAgent, self ).__init__()

        self._name = name
        self._config = config
        self._model = model
        self._seed = 0
        self._mode = 'train'


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
            return self._model.eval( state[np.newaxis,...] )[0] # remove batch dimension


    def seed( self, seed = 0 ) :
        r"""Seeds the agent with a given seed

        Args:
            seed (int): seed for the random number generator

        """
        if not self._model :
            print( 'ERROR> this agent has no model' )
        else :
            self._model.seed( seed )


    def setMode( self, mode ) :
        r"""Sets the mode the agent is going to be used (train|test)

        Args:
            mode (str): mode (train|test|...) which the agent will be set to

        """
        self._mode = mode


    def getMode( self ) :
        r"""Gets the current mode of the agent

        Returns:
            (str): Current mode of the agent (train|test|...)

        """
        return self._mode


    @abc.abstractmethod
    def update( self, transition ) :
        r"""Updates agent internals when a step is made

        Args:
            transition (tuple): a transition tuple ( s, a, r, s', done )

        """
        pass


    def clone( self, other ) :
        r"""Copies the whole model weights from another given model
        
        Args:
            other (PGModel): model from which to copy the weights

        """
        if not self._model :
            print( 'ERROR> this agent has no model' )
        else :
            self._model.clone( other._model )

import random
import numpy as np

from collections import namedtuple
from collections import deque

from IPython.core.debugger import set_trace

# Helper functionality for the dqn agent #######################################

class ReplayBuffer( object ) :

    def __init__( self, bufferSize, randomSeed ) :
        super( ReplayBuffer, self ).__init__()

        self._bufferSize = bufferSize
        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )

        self._memory = deque( maxlen = bufferSize )
        self._randomState = random.seed( randomSeed )

    def add( self, state, action, reward, nextState, endFlag ) :
        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )
        # and add it to the deque memory
        self._memory.append( _expObj )

    def sample( self ) :
        # grab a batch from the deque memory
        _expBatch = random.sample( self._memory, self._batchSize )

        # stack each experience component along batch axis
        _states = np.stack( [ _exp.state for _exp in _expBatch if _exp is not None ] )
        _actions = np.stack( [ _exp.action for _exp in _expBatch if _exp is not None ] )
        _rewards = np.stack( [ _exp.reward for _exp in _expBatch if _exp is not None ] )
        _nextStates = np.stack( [ _exp.nextState for _exp in _expBatch if _exp is not None ] )
        _endFlags = np.stack( [ _exp.endFlag for _exp in _expBatch if _exp is not None ] ).astype( np.uint8 )

        return _states, _actions, _rewards, _nextStates, _endFlags

    def __len__( self ) :
        return len( self._memory )


class DqnAgentConfig( object ) :

    def __init__( self ) :
        super( DqnConfig, self ).__init__()

        # environment state and action info
        self.stateDim = 7056
        self.nActions = 18

        # parameters for linear schedule of eps
        self.epsilonStart       = 1.0
        self.epsilonEnd         = 0.1
        self.epsilonSteps       = 100000
        self.epsilonDecay       = 0.995
        self.epsilonSchedule    = 'linear'

        # learning rate and related parameters
        self.lr                         = 0.00025
        self.minibatchSize              = 32
        self.learningStartsAt           = 50000
        self.learningUpdateFreq         = 4
        self.learningUpdateTargetFreq   = 10000
        self.learningMaxSteps           = 50000000

        # size of replay buffer
        self.replayBufferSize = 1000000

        # discount factor
        self.discount = 0.99

        # random seed
        self.seed = 1

    def save( self, filename ) :
        pass

    def load( self, filename ) :
        pass

class DqnModelConfig( object ) :

    def __init__( self ) :
        super( DqnModelConfig, self ).__init__()

        # shape of the input tensor for the model
        self.inputShape = ( 4, 84, 84 )
        self.outputShape = ( 18 )
        self.layers = [ { 'type' : 'conv2d', 'ksize' : 8, 'kstride' : 4, 'nfilters' : 32, 'activation' : 'relu' },
                        { 'type' : 'conv2d', 'ksize' : 4, 'kstride' : 2, 'nfilters' : 64, 'activation' : 'relu' },
                        { 'type' : 'conv2d', 'ksize' : 3, 'kstride' : 1, 'nfilters' : 64, 'activation' : 'relu' },
                        { 'type' : 'flatten' },
                        { 'type' : 'fc', 'units' : 512, 'activation' : 'relu' },
                        { 'type' : 'fc', 'units' : 18, 'activation' : 'relu' } ]

        # parameters copied from the agent configuration
        self._lr = 0.00025

    def save( self, filename ) :
        pass

    def load( self, filename ) :
        pass

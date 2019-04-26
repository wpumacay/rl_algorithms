
import numpy as np

# our helpers
from rl.dqn.utils import dqn_utils

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnAgent( object ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder ) :
        """Constructs a generic Dqn agent, given configuration information

        Args:
            agentConfig (DqnAgentConfig) : config object with agent parameters
            modelConfig (DqnModelConfig) : config object with model parameters
            modelBuilder (IDqnModel) : factory function to instantiate the model

        """

        super( IDqnAgent, self ).__init__()

        # environment state and action spaces info
        self._stateDim = agentConfig.stateDims
        self._nActions = agentConfig.nActions

        # random seed
        self._seed = agentConfig.seed

        # parameters for linear schedule of eps
        self._epsStart      = agentConfig.epsilonStart
        self._epsEnd        = agentConfig.epsilonEnd
        self._epsSteps      = agentConfig.epsilonSteps
        self._epsDecay      = agentConfig.epsilonDecay
        self._epsSchedule   = agentConfig.epsilonSchedule
        self._epsilon       = self._epsStart

        # learning rate and related parameters
        self._lr                        = agentConfig.lr
        self._minibatchSize             = agentConfig.minibatchSize
        self._learningStartsAt          = agentConfig.learningStartsAt
        self._learningUpdateFreq        = agentConfig.learningUpdateFreq
        self._learningUpdateTargetFreq  = agentConfig.learningUpdateTargetFreq
        self._learningMaxSteps          = agentConfig.learningMaxSteps

        # size of replay buffer
        self._replayBufferSize = agentConfig.replayBufferSize

        # discount factor gamma
        self._gamma = agentConfig.discount

        # tau factor for soft-updates
        self._tau = agentConfig.tau

        # some counters used by the agent's logic
        self._istep = 0
        self._iepisode = 0

        # copy some parameters from the agent config into the model config
        modelConfig._lr = self._lr

        # create the model accordingly
        self._qnetwork_actor = modelBuilder( modelConfig )
        self._qnetwork_target = modelBuilder( modelConfig )
        self._qnetwork_target.clone( self._qnetwork_actor, tau = 1.0 )

        # replay buffer
        self._rbuffer = dqn_utils.ReplayBuffer( self._replayBufferSize,
                                                self._seed )

        # states (current and next) for the model representation
        self._currState = None
        self._nextState = None

    def act( self, state, inference = False ) :
        _qvalues = self._qnetwork_actor.eval( state )

        if inference :
            return np.argmax( _qvalues )
        else :
            return self._egreedy( _qvalues )

    def step( self, transition ) :
        # grab information from this transition
        _s, _a, _snext, _r, _done = transition
        # preprocess the raw state
        self._nextState = self._preprocess( _snext )
        if self._currState is None :
            self._currState = self._preprocess( _s ) # for first step
        # store in replay buffer
        self._rbuffer.add( self._currState, _a, self._nextState, _r, _done )

        # check if can do a training step
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateFreq == 0 :
            self._learn()

        # update the weights of the target network (every update_target steps)
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateTargetFreq == 0 :
           self._qnetwork_target.clone( self._qnetwork_actor, tau = self._tau )

        # save next state (where we currently are in the environment) as current
        self._currState = self._nextState

        # update the agent's step counter
        self._istep += 1
        # and the episode counter if we finished an episode
        if _done :
            self._iepisode += 1

        # check epsilon update schedule and update accordingly
        if self._epsSchedule == 'linear' :
            # update epsilon using linear schedule
            _epsFactor = 1. - ( max( 0, self._istep - self._learningStartsAt ) / self._epsSteps )
            _epsDelta = max( 0, ( self._epsStart - self._epsEnd ) * _epsFactor )
            self._epsilon = self._epsEnd + _epsDelta

        elif self._epsSchedule == 'geometric' :
            # update epsilon with a geometric decay given by a decay factor
            _epsFactor = self._epsDecay if self._istep >= self._learningStartsAt else 1.0
            self._epsilon = max( self._epsEnd, self._epsilon * _epsFactor )

    def _egreedy( self, qvalues ) :
        """Get the action to take using eps-greedy over the given qvalues

        Args:
            qvalues (np.ndarray) : q-values evaluated from the model

        Returns:
            int : action to take using eps-greedy approach

        """

        # give all actions some small exploratory prob -> eps / nActions
        _probs = np.ones( self._nActions ) * self._epsilon / self._nActions
        _greedyAction = np.argmax( qvalues )
        # give greedy action some bigger prob -> 1 - eps + eps/nA
        _probs[ _greedyAction ] += 1.0 - self._epsilon
        # normalize just in case
        _probs /= np.sum( _probs )

        return np.random.choice( self._nActions, p = _probs )

    def _preprocess( self, rawState ) :
        """Preprocess a raw state into an appropriate state representation
    
        Args:
            rawState (np.ndarray) : raw state to be transformed

        Returns:
            np.ndarray : preprocess state into the approrpiate representation
        """

        """ OVERRIDE this method with your specific preprocessing """

        raise NotImplementedError( 'IDqnAgent::_preprocess> virtual method' )
        
    def _learn( self ) :
        """Makes a learning step using the DQN algorithm from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """

        # get a minibatch from the replay buffer
        _minibatch = self._rbuffer.sample( self._minibatchSize )
        _states, _actions, _rewards, _nextStates, _dones = _minibatch

        # compute targets (in a vectorized way). Recall:
        #               
        #            |->   _reward                            if s' is a terminal
        # q-target = |     
        #            |->   _reward + gamma * max( Q(s',a') )  otherwise
        #                                     a'
        # Or in vectorized form ( recall Q(s') computes all qvalues ) :
        #
        # qtargets = _rewards + (1 - terminals) * gamma * max(Q(nextStates), batchAxis)
        #
        # Notes: 
        #       * Just to clarify, we are assuming that in this call to Q
        #         the targets generated are not dependent of the weights
        #         of the network (should not take into consideration gradients 
        #         here, nor take them as part of the computation graph).
        #         Basically the targets are like training data from a 'dataset'.

        _qtargets = _rewards + ( 1 - _dones ) * self._gamma * \
                    self._qnetwork_target.eval( _nextStates )

        # make the learning call to the model (kind of like supervised setting)
        self._qnetwork_actor.train( _states, _qtargets )



import numpy as np
import dis_utils as utils

from IPython.core.debugger import set_trace


class QLearningDiscretizationAgent( object ) :

    def __init__( self, sLow, sHigh, nActions, gamma, epsilon, alpha, useAlphaDecay = False, useEpsilonDecay = False ):
        super( QLearningDiscretizationAgent, self ).__init__()

        # sanity check: should have both len = ndim state space
        assert len( sLow ) == len( sHigh ) , 'ERROR> low-high dimensions mismatch'

        # state space parameters
        self._slow = sLow
        self._shigh = sHigh
        self._dimS = len( sLow )

        # number of action (action space is assumed to be discrete)
        self._nA = nActions

        # discount factor
        self._gamma = gamma

        # alpha hyperparameter and decay params
        self._startEpsilon = epsilon
        self._endEpsilon = 0.01
        self._epsilonUseDecay = useEpsilonDecay
        self._epsilonDecay = 0.9995
        self._epsilon = epsilon

        # alpha hyperparameter and decay params
        self._startAlpha = alpha
        self._endAlpha = 0.01
        self._alphaUseDecay = useAlphaDecay
        self._alphaDecay = 0.99999
        self._alpha = alpha

        # episode counter (for 1/t decay schedule)
        self._iepisode = 1

        # model used for the q function (grid or tiling)
        self._qfunction = None

    def save( self, filename ) :
        if self._qfunction :
            self._qfunction.save( filename )

    def load( self, filename ) :
        if self._qfunction :
            self._qfunction.load( filename )

    def _epsGreedyActDistribution( self, state ) :
        # non greedy actions have equal prob. eps/nA
        _probs = np.ones( self._nA ) * self._epsilon / self._nA

        if self._qfunction :
            _qValues = [ self._qfunction.eval( state, action ) for action in range( self._nA ) ]
            _greedyAction = np.argmax( _qValues )
            # greedy action has prob 1 - eps + eps/nA
            _probs[ _greedyAction ] += 1.0 - self._epsilon
            # normalize just in case
            _probs = _probs / np.sum( _probs )

        return _probs

    def _epsGreedyAct( self, state ) :
        # get the current distribution over action for the e-greedy policy
        _aprobs = self._epsGreedyActDistribution( state )

        return np.random.choice( self._nA, p = _aprobs )

    def act( self, state, inference = False ) :
        if inference :
            _qValues = [ self._qfunction.eval( state, action ) for action in range( self._nA ) ]
            return np.argmax( _qValues )
        else :
            return self._epsGreedyAct( state )

    def onEndEpisode( self ) :
        self._iepisode += 1

        if self._epsilonUseDecay :
            # decrease epsilon by a factor every time step
            self._epsilon = max( self._endEpsilon, self._epsilon * self._epsilonDecay )
        else :
            # decrease epsilon using a 1/t schedule
            self._epsilon = max( 1 / self._iepisode, self._endEpsilon )

        # decay alpha (if given)
        if self._alphaUseDecay :
            self._alpha = max( self._endAlpha, self._alphaDecay * self._alpha )

    def reset( self ) :
        self._epsilon = self._startEpsilon
        self._alpha = self._startAlpha
        self._iepisode = 1

    def update( self, transition ) :

        if self._qfunction :
            _s, _a, _r, _snext, _done = transition
    
            # compute td-target (estimate of the return) similar to DQN part, in ...
            # which at termination steps the estimated return is the reward for the step
            if _done :
                _qTarget = _r
            else :
                _qValues = [ self._qfunction.eval( _snext, _a ) for action in range( self._nA ) ]
                _qTarget = _r + self._gamma * np.max( _qValues )
    
            ## set_trace()

            # update the q-value towards this estimate
            self._qfunction.update( _s, _a, _qTarget, self._alpha )

    @property
    def qfunction( self ) :
        return self._qfunction

    @property
    def dimS( self ) :
        return self._dimS

    @property    
    def nA( self ) :
        return self._nA

    @property
    def gamma( self ) :
        return self._gamma

    @property
    def epsilon( self ) :
        return self._epsilon

    @property
    def alpha( self ) :
        return self._alpha


class QLearningGridAgent( QLearningDiscretizationAgent ) :

    def __init__( self, sLow, sHigh, nActions, nBins, gamma, epsilon, alpha, useAlphaDecay = False, useEpsilonDecay = False ):
        super( QLearningGridAgent, self ).__init__( sLow, sHigh, nActions, gamma, epsilon, alpha, useAlphaDecay, useEpsilonDecay )

        # number of bins in which the state space is going to be partitioned
        self._nBins = nBins

        # create a grid over the state space
        self._grid = utils.createGrid( sLow, sHigh, nBins )

        # create the qfunction for this representation
        self._qfunction = utils.QFunctionGridTable( self._grid, self._nA )

        # some logs to check everything is ok
        print( "State space size:", tuple( len(splits) + 1 for splits in self._grid ) )
        print( "Q table  size:", self._qfunction.table.shape )

class QLearningTilingAgent( QLearningDiscretizationAgent ) :

    def __init__( self, sLow, sHigh, nActions, tilingSpecs, gamma, epsilon, alpha, useAlphaDecay = False, useEpsilonDecay = False ):
        super( QLearningTilingAgent, self ).__init__( sLow, sHigh, nActions, gamma, epsilon, alpha, useAlphaDecay, useEpsilonDecay )

        # specs to be used for the tilings of the state space
        self._tilingSpecs = tilingSpecs

        # create the tilings over the state space according to the specs given
        self._tilings = utils.createTilings( sLow, sHigh, tilingSpecs )

        # create the qfunction for this representation
        self._qfunction = utils.QFunctionTilingTable( self._tilings, self._nA )


if __name__ == '__main__' :
    # just a sinple test for the grid agent
    _agent1 = QLearningGridAgent( [-1,-5], [1,5], 2, (10, 10), 0.99, 1.0, 0.1, False, False )

    # and for the tiling agent as well
    _tilingSpecs = [ ( ( 10, 10 ), ( -0.066, -0.33 ) ),
                     ( ( 10, 10 ), ( 0.0, 0.0 ) ),
                     ( ( 10, 10 ), ( 0.066, 0.33 ) ) ]
    _agent2 = QLearningTilingAgent( [-1,-5], [1,5], 2, _tilingSpecs, 0.99, 1.0, 0.1, False, False )

    print( 'FILE IS OK!' )
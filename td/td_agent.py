
import pickle
import numpy as np


class ITDAgent( object ) :

    def __init__( self ) :
        super( ITDAgent, self ).__init__()

    def update( self, transition ) :
        raise NotImplementedError( 'ITDAgent::update> virtual method' )

    def act( self, state, inference = False ) :
        raise NotImplementedError( 'ITDAgent::act> virtual method' )

    def reset( self ) :
        raise NotImplementedError( 'ITDAgent::reset> virtual method' )

class ITDAgentDiscrete( ITDAgent ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha, useAlphaDecay = False ) :
        super( ITDAgentDiscrete, self ).__init__()

        self._nS = nS
        self._nA = nA
        self._gamma = gamma

        self._startEpsilon = epsilon
        self._endEpsilon = 0.001
        self._epsilonDecay = 0.999999
        self._epsilon = epsilon

        self._startAlpha = alpha
        self._endAlpha = 0.001
        self._alphaUseDecay = useAlphaDecay
        self._alphaDecay = 0.9999
        self._alpha = alpha

        self._vTable = np.zeros( ( nS ), dtype = np.float64 )
        self._qTable = np.zeros( ( nS, nA ), dtype = np.float64 )

    def save( self, filename ) :
        _data = { 'vTable' : self._vTable,
                  'qTable' : self._qTable }

        with open( filename, 'wb' ) as fh :
            pickle.dump( _data, fh )

    def load( self, filename ) :
        with open( filename, 'rb' ) as fh :
            _data = pickle.load( fh )

        self._vTable = _data['vTable']
        self._qTable = _data['qTable']

    def _epsGreedyAct( self, state ) :
        # non greedy actions have equal prob. eps/nA
        _probs = np.ones( self._nA ) * self._epsilon / self._nA
        # greedy action has prob 1 - eps + eps/nA
        _greedyAction = np.argmax( self._qTable[state] )
        _probs[ _greedyAction ] += 1.0 - self._epsilon
        # normalize just in case
        _probs = _probs / np.sum( _probs )

        # decrease epsilon using a 1/t schedule
        self._epsilon = max( self._endEpsilon, self._epsilon * self._epsilonDecay )

        return np.random.choice( self._nA, p = _probs )

    def act( self, state, inference = False ) :
        if inference :
            return np.argmax( self._qTable[state] )
        else :
            return self._epsGreedyAct( state )

    def reset( self ) :
        self._epsilon = self._startEpsilon
        self._alpha = self._startAlpha
        self._iepisode = 0

    def V( self ) :
        return self._vTable

    def Q( self ) :
        return self._qTable

    def epsilon( self ) :
        return self._epsilon

    def alpha( self ) :
        return self._alpha

class TDPredictionAgent( ITDAgentDiscrete ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha, useAlphaDecay = False ) :
        super( TDPredictionAgent, self ).__init__( nS, nA, gamma, epsilon, alpha, useAlphaDecay )

    def update( self, transition ) :
        _s, _a, _r, _snext, _done = transition

        # compute td-target (estimate of the return)
        if _done :
            _tdTarget = _r
        else :
            _tdTarget = _r + self._gamma * self._vTable[_snext]

        ## # compute td-target (estimate of the return)
        ## _tdTarget = _r + self._gamma * self._vTable[_snext]

        # update the v-value towards this estimate
        self._vTable[_s] = self._vTable[_s] + self._alpha * ( _tdTarget - self._vTable[_s] )

        # decay alpha (if given)
        if self._alphaUseDecay :
            self._alpha = max( self._endAlpha, self._alphaDecay * self._alpha )

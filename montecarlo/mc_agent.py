
import numpy as np
from collections import defaultdict


class MCAgent( object ):

    def __init__( self ) :
        super( MCAgent, self ).__init__()

    def beginEpisode( self ) :
        raise NotImplementedError( "MCAgent::begin> virtual method" )

    def update( self, info ) :
        raise NotImplementedError( "MCAgent::update> virtual method" )

    def endEpisode( self, info ) :
        raise NotImplementedError( "MCAgent::end> virtual method" )

    def act( self, state ) :
        raise NotImplementedError( "MCAgent::act> virtual method" )

    def reset( self ) :
        pass # nothing for now

    def computeReturnsToGo( self, rewards, gamma ) :
        # compute discounts to go
        _discs = np.array( [ gamma ** t for t in range( len( rewards ) ) ] )
        # compute returns to go
        _returnsToGo = []
        for i in range( len( _discs ) ) :
            _G = 0.0
            if i == 0 :
                _G = np.sum( rewards[:] * _discs[:] )
            else :
                _G = np.sum( rewards[i:] * _discs[:-i] )

            _returnsToGo.append( _G )

        return np.array( _returnsToGo )


class MCAgentDiscrete( MCAgent ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha = None ) :
        super( MCAgentDiscrete, self ).__init__()

        self.m_nS = nS
        self.m_nA = nA
        self.m_gamma = gamma

        self.m_startEpsilon = epsilon
        self.m_endEpsilon = 0.0
        self.m_epsilonDecay = 0.99999
        self.m_epsilon = epsilon

        self.m_alpha = alpha

        self.m_vTable = defaultdict( lambda: 0.0 )
        self.m_vNcount = defaultdict( lambda: 0 )
        self.m_vGret   = defaultdict( lambda: 0.0 )

        self.m_qTable = defaultdict( lambda: np.zeros( nA ) )
        self.m_qNcount = defaultdict( lambda: np.zeros( nA ) )
        self.m_qGret = defaultdict( lambda: np.zeros( nA ) )

    def beginEpisode( self ) :
        pass # just do nothing

    def update( self, info ) :
        pass # all computations are made at the end of an episode

    def _epsGreedyAct( self, state ) :
        # non greedy actions have equal prob. eps/nA
        _probs = np.ones( self.m_nA ) * self.m_epsilon / self.m_nA
        # greedy action has prob 1 - eps + eps/nA
        _greedyAction = np.argmax( self.m_qTable[state] )
        _probs[ _greedyAction ] += 1.0 - self.m_epsilon
        # normalize just in case
        _probs = _probs / np.sum( _probs )

        # decrease epsilon using a 1/t schedule
        self.m_epsilon = max( self.m_endEpsilon, self.m_epsilon * self.m_epsilonDecay )

        return np.random.choice( self.m_nA, p = _probs )

    def act( self, state, inference ) :
        if inference :
            return np.argmax( self.m_qTable[state] )
        else :
            return self._epsGreedyAct( state )

    def reset( self ) :
        self.m_epsilon = self.m_startEpsilon

    def V( self ) :
        return self.m_vTable

    def Q( self ) :
        return self.m_qTable

    def stateVisits( self ) :
        return self.m_vNcount

    def stateActionVisits( self ) :
        return self.m_qNcount

    def epsilon( self ) :
        return self.m_epsilon

class MCAgentDiscreteFirstVisit( MCAgentDiscrete ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha = None ) :
        super( MCAgentDiscreteFirstVisit, self ).__init__( nS, nA, gamma, epsilon, alpha )

    def endEpisode( self, info ) :
        # sanity check
        assert ( 'episode' in info ), 'Must pass episode in info dict'

        # grabt the episode information
        _episode = info['episode']
        # and extract the rewards list
        _, _, _rewards = zip( *_episode )
        _rewards = np.array( _rewards )
        # compute the returns to go
        _returns = self.computeReturnsToGo( _rewards, self.m_gamma )
        # and some checking for first-visit
        _visitedS = defaultdict( lambda: False )
        _visitedSA = defaultdict( lambda: np.zeros( self.m_nA, dtype = np.bool ) )

        for i in range( len( _episode ) ) :
            _G = _returns[i]
            _s = _episode[i][0]
            _a = _episode[i][1]

            # check for first-visit (state only) ###############################
            if _visitedS[_s] :
                continue

            # cache the visit
            _visitedS[_s] = True
            # update counters
            self.m_vNcount[_s] += 1
            self.m_vGret[_s] += _G

            # update the v-table
            if self.m_alpha is None :
                ## self.m_vTable[_s] += ( 1. / self.m_vNcount[_s] ) * ( _G - self.m_vTable[_s] )
                self.m_vTable[_s] = self.m_vGret[_s] / self.m_vNcount[_s]
            else :
                self.m_vTable[_s] += self.m_alpha * ( _G - self.m_vTable[_s] )

            # ##################################################################

            # check for first-visit (state-action) #############################
            if _visitedSA[_s][_a]:
                continue

            # cache the visit
            _visitedSA[_s][_a] = True
            # update counters
            self.m_qNcount[_s][_a] += 1
            self.m_qGret[_s][_a] += _G

            # update the q-table
            if self.m_alpha is None :
                self.m_qTable[_s][_a] = self.m_qGret[_s][_a] / self.m_qNcount[_s][_a]
            else :
                self.m_qTable[_s][_a] += self.m_alpha * ( _G - self.m_qTable[_s][_a] )

            # ##################################################################


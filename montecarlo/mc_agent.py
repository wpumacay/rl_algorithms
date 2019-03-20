
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


class MCAgentDiscrete( MCAgent ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha = None ) :
        super( MCAgentDiscrete, self ).__init__()

        self.m_nS = nS
        self.m_nA = nA
        self.m_gamma = gamma
        self.m_epsilon = epsilon
        self.m_alpha = alpha

        self.m_vtable = defaultdict( lambda: 0.0 )
        self.m_qtable = defaultdict( lambda: np.zeros( nA ) )
        self.m_ncount = defaultdict( lambda: 0 )
        self.m_gret   = defaultdict( lambda: 0 )

    def beginEpisode( self ) :
        pass # just do nothing

    def update( self, info ) :
        pass # all computations are made at the end of an episode

    def _egreedyAct( self, state ) :
        # non greedy actions have equal prob. eps/nA
        _probs = np.ones( self.m_nA ) / self.m_nA
        # greedy action has prob 1 - eps + eps/nA
        _greedyAction = np.argmax( self.m_qtable[state] )
        _probs[ _greedyAction ] += 1.0 - self.m_epsilon

        return np.random.choice( self.m_nA, p = _probs )

    def act( self, state ) :
        return np.argmax( self.m_qtable[state] )

    def V( self ) :
        return self.m_vtable

    def Q( self ) :
        return self.m_qtable


class MCAgentDiscreteFirstVisit( MCAgentDiscrete ) :

    def __init__( self, nS, nA, gamma, epsilon, alpha = None ) :
        super( MCAgentDiscreteFirstVisit, self ).__init__( nS, nA, gamma, epsilon, alpha )

    def _mcPredictionFirstVisitV( self, info ) :
        # sanity check
        assert ( 'episode' in info ), 'Must pass episode in info dict'

        _G = 0
        _episode = info['episode']
        _episode.reverse()
        _visited = {}

        for _s, _a, _r in _episode :
            _G = _r + self.m_gamma * _G
            if _s not in _visited :
                _visited[_s] = True
                self.m_ncount[_s] += 1
                self.m_gret[_s] += _G
                if self.m_alpha is None :
                    ## self.m_vtable[_s] += ( 1. / self.m_ncount[_s] ) * ( _G - self.m_vtable[_s] )
                    self.m_vtable[_s] = self.m_gret[_s] / self.m_ncount[_s]
                else :
                    self.m_vtable[_s] += self.m_alpha * ( _G - self.m_vtable[_s] )

    def _mcControlFirstVisitQ( self, info ) :
        pass

    def endEpisode( self, info ):
        # MC-First visit for Vpi
        self._mcPredictionFirstVisitV( info )
        # MC-First visit for Q*
        self._mcControlFirstVisitQ( info )
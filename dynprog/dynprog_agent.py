
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class IDynProgAgent( object ) :

    def __init__( self, model, nS, nA ) :
        super( IDynProgAgent, self ).__init__()

        # dynamics of the environment (should be in the form { s: { a: [()] } })
        self._p = model
        # dimensions of state and action spaces
        self._nS = nS
        self._nA = nA
        # V table
        self._v = defaultdict( lambda : 0.0 )
        # Q table
        self._q = defaultdict( lambda : np.zeros( nA ) )

    @property
    def v( self ) :
        return self._v
    
    @property
    def q( self ) :
        return self._q

    def act( self, state ) :
        raise NotImplementedError( 'IDynProgAgent::act> pure virtual' )

    def run( self ) :
        raise NotImplementedError( 'IDynProgAgent::learn> pure virtual' )


class PolicyEvalAgent( IDynProgAgent ) :

    def __init__( self, env, nS, nA, gamma, policyFcn ) :
        super( PolicyEvalAgent, self ).__init__( env.P, nS, nA )

        # hardcoded policy given by the user
        self._policyFcn = policyFcn

        # environment itself
        self._env = env

        # discount factor
        self._gamma = gamma

    def _bellmanExpBackup( self, state, action, poutcomes ) :
        pass

    def run( self ) :

        for _ in tqdm( range( 100000 ) ) :
            _vnew = defaultdict( lambda : 0.0 )
            for s in range( self._nS ) :
                # run sthocastic policy given by user
                _aprobs = self._policyFcn( self._env, s )
                # prepare for backup over all actions
                for a in range( self._nA ) :
                    # grab the probability of this action given by the policy
                    _probA = _aprobs[a]
                    # grab step for this transition (s,a)
                    _transitions = self._p[s][a]
                    # do a backup with this transitions
                    for _prob, _snext, _reward, _ in _transitions :
                        _vnew[s] += _probA * _prob * ( _reward + self._gamma * self._v[_snext] )

            self._v = _vnew





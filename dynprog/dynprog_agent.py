
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class IDynProgAgent( object ) :

    def __init__( self, model, nS, nA, gamma ) :
        super( IDynProgAgent, self ).__init__()

        # dynamics of the environment (should be in the form { s: { a: [()] } })
        self._p = model
        # dimensions of state and action spaces
        self._nS = nS
        self._nA = nA
        # discount factor
        self._gamma = gamma
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

    def _recoverQfromV( self ) :
        ## Computing Q from V
        _qnew = defaultdict( lambda : np.zeros( self._nA ) )
        for s in range( self._nS ) :
            for a in range( self._nA ) :
                # grab step for this transition (s,a)
                _transitions = self._p[s][a]
                for _prob, _snext, _reward, _ in _transitions :
                    _qnew[s][a] += _prob * ( _reward + self._gamma * self._v[_snext] )

        self._q = _qnew


class PolicyEvalAgent( IDynProgAgent ) :

    def __init__( self, env, nS, nA, gamma, policyFcn ) :
        super( PolicyEvalAgent, self ).__init__( env.P, nS, nA, gamma )

        # hardcoded policy given by the user
        self._policyFcn = policyFcn
        # environment itself
        self._env = env
        # convergence tolerance
        self._thresh = 1e-3

    def run( self ) :

        while True :
            _maxDiffV = 0.0
            # policy evaluation for state-value function V(s)
            #                                               
            #             s         V(s)                                        
            #            / \                                                    
            #           /   \     pi(a|s)                                       
            #          /     \                                                  
            #         a  ...  a                                                 
            #        / \     / \     p(s',r|s,a)                                          
            #       /   \   /   \                                               
            #      s'    ...     s' V(s')                                       
            #                                                                   
            #                                                                   
            #                                                                   
            #        ---          ---                                           
            # V(s) = \            \                                             
            #  k     /    pi(a|s) /     p(s',r|s,a)( r + gamma * V(s') )        
            #        ---          ---                             k-1              
            #         a           s', r                                         
            #         
            _vnew = defaultdict( lambda : 0.0 )
            for s in range( self._nS ) :
                # run stochastic policy given by user
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

                _maxDiffV = max( _maxDiffV, np.abs( _vnew[s] - self._v[s] ) )

            self._v = _vnew

            # check if converged
            if _maxDiffV < self._thresh :
                break

        # recover the q-function from the v-function
        self._recoverQfromV()

        ## Another way of computing Q (running policy evaluation on Q itself)
        ## It works, but for evaluation it is O(|S|^2 |A|^2), which is more ...
        ## exponsive if a model is given (V evaluation is O(|S|^2 |A|))

##             # policy evaluation for action-value function Q(s,a)
##             #                                               
##             #           (s,a)          Q(s,a)                                        
##             #            / \                                                    
##             #           /   \        p(s',r|s,a)                                       
##             #          /     \                                                  
##             #         s'  ...  s'                                                 
##             #        / \     / \      pi(a'|s')                                          
##             #       /   \   /   \                                               
##             #      a'    ...     a'   Q(s',a')                                       
##             #                                                                   
##             #                                                                   
##             #                                                                   
##             #          ---                            ---                         
##             # Q(s,a) = \                              \                           
##             #  k       /     p(s',r|s,a)( r + gamma * /    pi(a|s) Q(s',a') )        
##             #          ---                            ---           k-1              
##             #          s', r                           a                          
##             #         
##             _qnew = defaultdict( lambda : np.zeros( self._nA ) )
##             for s in range( self._nS ) :
##                 for a in range( self._nA ) :
##                     # grab step for this transition (s,a)
##                     _transitions = self._p[s][a]
##                     for _prob, _snext, _reward, _ in _transitions :
##                         # do a small part of the bellman backup
##                         _qnew[s][a] += _prob * _reward
##                         # run stochastic policy given by the user
##                         _aprobs = self._policyFcn( self._env, _snext )
##                         # prepare for backup over all actions
##                         for _anext in range( self._nA ) :
##                             _probAnext = _aprobs[_anext]
##                             # do the rest of the bellman backup
##                             _qnew[s][a] += _prob * ( self._gamma * _probAnext * self._q[_snext][_anext] )
## 
##             self._q = _qnew


class PolicyIterationAgent( IDynProgAgent ) :

    def __init__( self, env, nS, nA, gamma ) :
        super( PolicyIterationAgent, self ).__init__( env.P, nS, nA, gamma )

        self._env = env
        self._thresh = 1e-5
        self._policy = np.ones( ( nS, nA ) ) / nA

    def _policyEvaluation( self ) :

        while True :

            _maxDeltaV = 0.0

            # policy evaluation for state-value function V(s)
            #                                               
            #             s         V(s)                                        
            #            / \                                                    
            #           /   \     pi(a|s)                                       
            #          /     \                                                  
            #         a  ...  a                                                 
            #        / \     / \     p(s',r|s,a)                                          
            #       /   \   /   \                                               
            #      s'    ...     s' V(s')                                       
            #                                                                   
            #                                                                   
            #                                                                   
            #        ---          ---                                           
            # V(s) = \            \                                             
            #  k     /    pi(a|s) /     p(s',r|s,a)( r + gamma * V(s') )        
            #        ---          ---                             k-1              
            #         a           s', r                                         
            #         
            _vnew = defaultdict( lambda : 0.0 )
            for s in range( self._nS ) :
                # run stochastic policy given by user
                _aprobs = self._policy[s]
                # prepare for backup over all actions
                for a in range( self._nA ) :
                    # grab the probability of this action given by the policy
                    _probA = _aprobs[a]
                    # grab step for this transition (s,a)
                    _transitions = self._p[s][a]
                    # do a backup with this transitions
                    for _prob, _snext, _reward, _ in _transitions :
                        _vnew[s] += _probA * _prob * ( _reward + self._gamma * self._v[_snext] )

                _maxDeltaV = max( _maxDeltaV, np.abs( _vnew[s] - self._v[s] ) )

            self._v = _vnew

            if _maxDeltaV < self._thresh:
                break

        ## Recover q-values from v (for easier improvement step)
        self._recoverQfromV()

    def _policyImprovement( self ) :
        for s in range( self._nS ) :
            _bestActionIndx = np.argmax( self._q[s] )
            for a in range( self._nA ) :
                self._policy[s][a] = 1.0 if a == _bestActionIndx else 0.0

    def policy( self ) :
        return self._policy

    def run( self ) :
        print( 'INFO> Started policy iteration' )
        for _ in tqdm( range( 1000 ) ) :
            
            self._policyEvaluation()
            self._policyImprovement()



class ValueIterationAgent( IDynProgAgent ) :

    def __init__( self, env, nS, nA, gamma ) :
        super( ValueIterationAgent, self ).__init__( env.P, nS, nA, gamma )

        self._env = env
        self._thresh = 1e-5
        self._policy = np.ones( ( nS, nA ) ) / nA

    def policy( self ) :
        return self._policy

    def run( self ) :
        while True :

            _maxDeltaV = 0.0

            # value iteration (to compute V*(s))
            #                                               
            #             s         V(s)                                        
            #            / \                                                    
            #           /   \       maximize over a                                       
            #          /     \                                                  
            #         a  ...  a                                                 
            #        / \     / \     p(s',r|s,a)                                          
            #       /   \   /   \                                               
            #      s'    ...     s' V(s')                                       
            #                                                                   
            #                                                                   
            #              ---                                                                                                      
            #              \                                             
            # V(s) = max   /     p(s',r|s,a)( r + gamma * V(s') )        
            #  k      a    ---                             k-1              
            #              s', r                                         
            #        
            #         
            _vnew = defaultdict( lambda : 0.0 )
            for s in range( self._nS ) :
                _vcandidates = []
                for a in range( self._nA ) :
                    # grab step for this transition (s,a)
                    _transitions = self._p[s][a]
                    # define the candidate value for this part of the backup
                    _vcandidate = 0.0
                    # do part of the backup with this transitions
                    for _prob, _snext, _reward, _ in _transitions :
                        _vcandidate += _prob * ( _reward + self._gamma * self._v[_snext] )
                    # store candidate for max-part of the backup
                    _vcandidates.append( _vcandidate )

                # compute the max-part of the backup
                _vnew[s] = max( _vcandidates )

                # compute diff to check for convergence
                _maxDeltaV = max( _maxDeltaV, np.abs( _vnew[s] - self._v[s] ) )

            self._v = _vnew

            if _maxDeltaV < self._thresh:
                break

        # recover q-function for later policy-computation
        self._recoverQfromV()
        
        # recover policy from q-function
        for s in range( self._nS ) :
            self._policy[s] = np.argmax( self._q[s] )
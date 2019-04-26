
from rl.dqn.core.dqn_model_interface import IDqnModel

import numpy as np



class DqnModelTable( IDqnModel ) :
    """A simple test model represented as a table (numpy array)

    """

    def __init__( self, name, modelConfig ) :
        super( DqnModelTable, self ).__init__( name, modelConfig )

    def build( self ) :
        self._qtable = np.zeros( self._inputShape + self._outputShape )

    def eval( self, state, inference = False ) :
        return self._qtable.take( state, axis = 0 )

    def train( self, states, actions, targets ) :

        # do forward pass to compute q-target predictions
        _qhat_s = self._qtable.take( states, axis = 0 )
        _qhat_sa = _qhat_s[np.arange( _qhat_s.shape[0]), actions ]

        # update using q-learning update rule (over batch)
        #   q(s,a) = q(s,a) + alpha * ( qtarget - q(s,a) )
        _qtable_new = self._qtable.copy()

        for s, a, qhat, qtarget in zip( states, actions, _qhat_sa, targets ) :
            _qtable_new[s,a] = qhat + self._lr * ( qtarget - qhat )

        self._qtable = _qtable_new
        
        # @TODO: Place here some history measurements

    def clone( self, other, tau = 1.0 ) :
        self._qtable = ( 1. - tau ) * self._qtable.copy() + ( tau ) * other._qtable.copy()

    def save( self, filename ) :
        pass

    def load( self, filename ) :
        pass

DqnModelBuilder = lambda name, config : DqnModelTable( name, config )
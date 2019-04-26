
import numpy as np

# our helpers
from rl.dqn.utils import dqn_utils

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnModel( object ) :

    def __init__( self, modelConfig ) :
        super( IDqnModel, self ).__init__()

        # save configuration data
        self._inputShape = modelConfig.inputShape
        self._outputShape = modelConfig.outputShape
        self._layersDefs = modelConfig.layers.copy()

        # save learning rate (copied from agent's configuration)
        self._lr = modelConfig._lr

        self.build()

    def build( self ) :
        raise NotImplementedError( 'IDqnModel::build> virtual method' )

    def eval( self, state ) :
        raise NotImplementedError( 'IDqnModel::eval> virtual method' )

    def train( self, states, actions, targets ) :
        raise NotImplementedError( 'IDqnModel::train> virtual method' )

    def clone( self, other, tau = 1.0 ) :
        raise NotImplementedError( 'IDqnModel::clone> virtual method' )

    def save( self, filename ) :
        raise NotImplementedError( 'IDqnModel::save> virtual method' )

    def load( self, filename ) :
        raise NotImplementedError( 'IDqnModel::load> virtual method' )


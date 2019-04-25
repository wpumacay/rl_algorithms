
import numpy as np

# our helpers
from . import dqn_utils

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnModel( object ) :

    def __init__( self, modelConfig ) :
        super( IDqnModel, self ).__init__()

        self._inputShape = modelConfig.inputShape.copy()
        self._outputShape = modelConfig.outputShape.copy()
        self._layersDefs = modelConfig.layers.copy()
        
        self._lr = modelConfig._lr

    def build( self ) :
        raise NotImplementedError( 'IDqnModel::build> virtual method' )

    def eval( self, state ) :
        raise NotImplementedError( 'IDqnModel::eval> virtual method' )

    def train( self, states, targets ) :
        raise NotImplementedError( 'IDqnModel::train> virtual method' )

    def save( self, filename ) :
        raise NotImplementedError( 'IDqnModel::save> virtual method' )

    def load( self, filename ) :
        raise NotImplementedError( 'IDqnModel::load> virtual method' )


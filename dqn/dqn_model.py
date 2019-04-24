
import numpy as np

# our helpers
import dqn_utils

# debugging helpers
from IPython.core.debugger import set_trace

class IDqnModel( object ) :

    def __init__( self, modelConfig ) :
        super( IDqnModel, self ).__init__()

        self._type = modelConfig.type
        self._inputShape = modelConfig.inputShape.copy()
        self._outputShape = modelConfig.outputShape.copy()
        self._layers = modelConfig.layers.copy()

        self._build()

    def _build( self ) :
        assert ( self._type == 'mlp' or self._type == 'cnn' ), \
               "ERROR> model type \'%s\' not supported" % ( self._type )

        if self._type == 'mlp' :
            self._buildMlpModel()
        elif self._type == 'cnn' :
            self._buildCnnModel()

    def _buildMlpModel( self ) :
        raise NotImplementedError( 'IDqnModel::_buildMlpModel> virtual method' )

    def _buildCnnModel( self ) :
        raise NotImplementedError( 'IDqnModel::_buildCnnModel> virtual method' )
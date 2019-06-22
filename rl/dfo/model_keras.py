
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras import optimizers

from rl.dfo.model import DFOModel

from IPython.core.debugger import set_trace

ERROR_MSG_MODEL_CREATION = 'ERROR> could not create model (%s)'
ERROR_MSG_LAYER_CREATION = 'ERROR> could not create layer (%s) of model (%s)'
ERROR_MSG_WRONG_PERTURBATION = 'ERROR> perturbation type (%s) is not supported'
ERROR_MSG_MODEL_MISMATCH = 'ERROR> there is a mismatch between model (%s) and (%s). Cannot copy'

class DFOModelKeras( DFOModel ) :

    def __init__( self, name, config ) :
        # forward declare some resources
        self._kerasBackboneModel = None

        super( DFOModelKeras, self ).__init__( name, config )


    def _buildModel( self ) :
        self._kerasBackboneModel = keras.Sequential()

        for _layerDef in self._layersDefs :
            _layer = self._buildLayer( _layerDef )

            assert _layer != None, \
                   ERROR_MSG_LAYER_CREATION % ( _layerDef['name'], self._name )

            self._backbone.append( _layer )
            self._kerasBackboneModel.add( _layer )


    def _buildLayer( self, layerDef ) :
        _layer = None
        _lname = layerDef['name']
        _ltype = layerDef['type']

        if _ltype == 'fc' :
            # units for the layer, and default as outputshape if last layer
            _lUnits = layerDef.get( 'units', self._outputShape[0] )
            # activation fcn for the layer, and default to linear if none given
            _lActivation = layerDef.get( 'activation', 'linear' )
            # whether or not to use bias
            _lUseBias = layerDef.get( 'useBias', True )
            # whether or not use an initializer
            _lInitializerId = layerDef.get( 'initializer', 'glorot_uniform' )
            _lInitializerArgs = layerDef.get( 'initializerArgs', {} )
            if _lInitializerId == 'uniform' :
                _lInitializer = initializers.RandomUniform( minval = _lInitializerArgs.get( 'min', -0.05 ),
                                                            maxval = _lInitializerArgs.get( 'max', 0.05 ),
                                                            seed = _lInitializerArgs.get( 'seed', None ) )
            else :
                _lInitializer = initializers.glorot_uniform( seed = _lInitializerArgs.get( 'seed', None ) )

            # create the dense|fully-connected layer
            if len( self._backbone ) < 1 :
                # first layer, from inputs to hidden (or perhaps output) units
                _layer = layers.Dense( units = _lUnits, 
                                       input_shape = self._inputShape,
                                       activation = _lActivation,
                                       use_bias = _lUseBias,
                                       kernel_initializer = _lInitializer )
            else :
                # intermediate layer, with input shape to be inferred by keras
                _layer = layers.Dense( units = _lUnits,
                                       activation = _lActivation,
                                       use_bias = _lUseBias,
                                       kernel_initializer = _lInitializer )

        elif _ltype == 'flatten' :
            # create a flatten layer (flatten n-dim cnn output volume to vector)
            pass
        elif _ltype == 'conv2d' :
            # create a conv2d layer
            pass

        return _layer


    def initialize( self, args = {} ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )

        # just compile the model, no session required, as it seems keras
        # handles this internally (perhaps a default graph session)

        _loss = losses.categorical_crossentropy if self._useDiscreteOutputs \
                    else losses.mean_squared_error

        # compile the model, but it actually does not need the losses, as we
        # might not need to run backprop through the model
        self._kerasBackboneModel.compile( loss = _loss,
                             optimizer = optimizers.Adam(),
                             metrics = ['accuracy'] )


    def eval( self, x ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )

        # call predict on x, which should have extra batch dimension
        return self._kerasBackboneModel.predict( x )


    def copy( self, other ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )

        if not other._kerasBackboneModel :
            return

        _srcWeights = self._kerasBackboneModel.get_weights()
        _dstWeights = other._kerasBackboneModel.get_weights()

        assert len( _srcWeights ) == len( _dstWeights ), ERROR_MSG_MODEL_MISMATCH \
                                                            % ( self._name, other._name )

        _weights = []
        for i in range( len( _dstWeights ) ) :
            _weights.append( _dstWeights[i].copy() )
            
        self._kerasBackboneModel.set_weights( _weights )


    def clone( self, name = None ) :
        # create a new model with the same configuration
        _clonedModel = DFOModelKeras( name if name is not None else ( self._name + '_clone' ),
                                      self._config )
        _clonedModel.initialize()
        # and copy the weights from this model into the cloned model
        _clonedModel.copy( self )

        return _clonedModel


    def perturb( self, ptype, args ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )
        assert ptype == 'uniform' or ptype == 'gaussian', ERROR_MSG_WRONG_PERTURBATION % ( ptype, )

        _weights = self._kerasBackboneModel.get_weights()
        for i in range( len( _weights ) ) :
            _perturbation = None

            if ptype == 'uniform' :
                _perturbationScale = args.get( 'perturbationScale', 1e-2 )
                _perturbation = _perturbationScale * np.random.rand( *_weights[i].shape )

            elif ptype == 'gaussian' :
                _perturbationSigma = args.get( 'perturbationSigma', 0.5 )
                _perturbationScale = args.get( 'perturbationScale', 1.0 )
                _perturbation = _perturbationSigma * _perturbationScale * \
                                np.random.randn( *_weights[i].shape )

            _weights[i] += _perturbation

        self._kerasBackboneModel.set_weights( _weights )


    def _printConfig( self ) :
        print( '#############################################################' )
        print( '#                                                           #' )
        print( '#                 Model configuration                       #' )
        print( '#                                                           #' )
        print( '#############################################################' )

        print( 'model name          : ', self._name )
        print( 'input shape         : ', self._inputShape )
        print( 'output shape        : ', self._outputShape )
        print( 'discrete outputs    : ', str( self._useDiscreteOutputs ) )

        print( 'model summary -----------------------------------------------' )
        self._kerasBackboneModel.summary()
        print( '-------------------------------------------------------------' )

        print( '#############################################################' )

import sys
import numpy as np

try :
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import losses
    from tensorflow.keras import initializers
    from tensorflow.keras import optimizers
except ImportError :
    print( 'ERROR> it seems you don\'t have keras installed in your system' )
    sys.exit( -1 )

from rl.dfo.model import DFOModel

from IPython.core.debugger import set_trace

ERROR_MSG_MODEL_CREATION = 'ERROR> could not create model (%s)'
ERROR_MSG_LAYER_CREATION = 'ERROR> could not create layer (%s) of model (%s)'
ERROR_MSG_WRONG_PERTURBATION = 'ERROR> perturbation type (%s) is not supported'
ERROR_MSG_MODEL_MISMATCH = 'ERROR> there is a mismatch between model (%s) and (%s). Cannot copy'

class DFOModelKeras( DFOModel ) :

    def __init__( self, name, config, verbose = True ) :
        # forward declare some resources
        self._kerasBackboneModel = None
        # random number generator
        self._randgen = None

        super( DFOModelKeras, self ).__init__( name, config, verbose )


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

        ## set_trace()

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
            elif _lInitializerId == 'normal' :
                _lInitializer = initializers.RandomNormal( mean = _lInitializerArgs.get( 'mean', 0.0 ),
                                                           stddev = _lInitializerArgs.get( 'stddev', 0.05 ),
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


    def seed( self, seed ) :
        # create and seed a random generator for this model to use
        self._randgen = np.random.RandomState( seed )


    def eval( self, x ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )

        ## set_trace()

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
                                      self._config,
                                      verbose = False )
        _clonedModel.initialize()
        # and copy the weights from this model into the cloned model
        _clonedModel.copy( self )

        return _clonedModel


    def weights( self ) :
        if self._kerasBackboneModel :
            return self._kerasBackboneModel.get_weights()

        return []


    def areWeightsEqual( self, otherWeights ) :
        if not self._kerasBackboneModel :
            return False

        _thisWeights = self._kerasBackboneModel.get_weights()

        # should have the same number of components (np.ndarrays)
        if len( _thisWeights ) != len( otherWeights ) :
            return False

        for _thisWeightsComp, _otherWeightsComp in zip( _thisWeights, otherWeights ) :
            # sanity check, both should be np.ndarrays
            assert isinstance( _thisWeightsComp, np.ndarray ) or \
                   isinstance( _otherWeightsComp, np.ndarray ) , \
                'ERROR> weights components to compare should be np.ndarrays'

            if _thisWeightsComp.shape != _otherWeightsComp.shape :
                return False

            if not np.array_equal( _thisWeightsComp, _otherWeightsComp ) :
                return False

        return True


    def perturb( self, ptype, args ) :
        assert self._kerasBackboneModel != None, ERROR_MSG_MODEL_CREATION % ( self._name, )
        assert ptype == 'uniform' or ptype == 'gaussian', ERROR_MSG_WRONG_PERTURBATION % ( ptype, )

        # check if the user wanted to recover the random generator to a given state
        _randState = args.get( 'randState', None )
        if _randState :
            self._randgen.set_state( _randState )

        # check if requesting only to compute perturbation but not applying it
        _applyPerturbation = args.get( 'applyPerturbation', True )

        # check if a randgen has been given
        _externRandGen = args.get( 'externRandGen', None )

        # grab state of the generator
        if _externRandGen :
            _befState = _externRandGen.get_state()
        else :
            _befState = self._randgen.get_state()

        _weights = self._kerasBackboneModel.get_weights()
        for i in range( len( _weights ) ) :
            _perturbation = None

            if ptype == 'uniform' :
                _perturbationScale = args.get( 'perturbationScale', 1e-2 )
                if _externRandGen :
                    _perturbation = _perturbationScale * _externRandGen.rand( *_weights[i].shape )
                else :
                    _perturbation = _perturbationScale * self._randgen.rand( *_weights[i].shape )

            elif ptype == 'gaussian' :
                _perturbationScale = args.get( 'perturbationScale', 1.0 )
                if _externRandGen :
                    _perturbation = _perturbationScale * _externRandGen.randn( *_weights[i].shape )
                else :
                    _perturbation = _perturbationScale * self._randgen.randn( *_weights[i].shape )

            if _applyPerturbation :
                _weights[i] += _perturbation

        if _applyPerturbation :
            self._kerasBackboneModel.set_weights( _weights )

        # grab state of the generator
        if _externRandGen :
            _nowState = _externRandGen.get_state()
        else :
            _nowState = self._randgen.get_state()

        return (_befState, _nowState)


    def save( self, filename ) :
        self._kerasBackboneModel.save_weights( filename + '.h5' )


    def load( self, filename ) :
        self._kerasBackboneModel.load_weights( filename + '.h5' )


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
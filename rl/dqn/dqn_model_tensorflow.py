
from rl.dqn.core.dqn_model_interface import IDqnModel

import numpy as np

from collections import deque

import tensorflow as tf
from tensorflow import keras

from IPython.core.debugger import set_trace

def createNetworkTestLunarLander( inputShape, outputShape, layersDefs ) :
    # lunar lander has a 8-vector as an observation (rank-1 tensor)
    assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
    # and also has a discrete set of actions, with a 4-vector for its qvalues
    assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

    # keep things simple (use keras for core model definition)
    _networkOps = keras.Sequential()

    # define initializers
    _kernelInitializer = keras.initializers.glorot_normal( seed = 0 )
    _biasInitializer = keras.initializers.Zeros()

    # add the layers for our test-case
    _networkOps.add( keras.layers.Dense( 128, activation = 'relu', input_shape = inputShape, kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 64, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 16, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( outputShape[0], kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )

    ## _networkOps.summary()

    return _networkOps

def createNetworkGeneric( inputShape, outputShape, layersDefs ) :
    pass

class DqnModelTensorflow( IDqnModel ) :

    def __init__( self, name, modelConfig ) :
        super( DqnModelTensorflow, self ).__init__( name, modelConfig )

        # to save the losses for later review
        self._losses = deque( maxlen = 100 )

    def build( self ) :
        # placeholders for taking data in
        self._tfStates              = tf.placeholder( tf.float32, (None,) + self._inputShape )
        self._tfActions             = tf.placeholder( tf.int32, (None,) )
        self._tfActionsIndices      = tf.placeholder( tf.int32, (None,) )
        self._tfQTargets            = tf.placeholder( tf.float32, (None,) )

        # create the nnetwork model architecture
        self._nnetwork = createNetworkTestLunarLander( self._inputShape,
                                                       self._outputShape,
                                                       self._layersDefs )
        
        # create the ops for evaluating the output of the model (Q(s,:))
        self._opQhat_s = self._nnetwork( self._tfStates )
        # @TODO|CHECK: Change the gather call by multiply + one-hot
        # create the ops for getting the Q(s,a) for each batch of (states) + (actions)
        # using tf.gather_nd, and expanding action indices with batch indices
        self._opActionsWithIndices = tf.stack( [self._tfActionsIndices, self._tfActions], axis = 1 )
        self._opQhat_sa = tf.gather_nd( self._opQhat_s, self._opActionsWithIndices )

        # create ops for the loss function
        self._opLoss = tf.losses.mean_squared_error( self._tfQTargets, self._opQhat_sa )

        # create ops for the loss and optimizer
        self._opOptim = tf.train.AdamOptimizer( learning_rate = self._lr ).minimize( self._opLoss, var_list = self._nnetwork.trainable_weights )

        # tf.Session, passed by the backend-initializer
        self._sess = None

    def initialize( self, args ) :
        # grab session and initialize
        self._sess = args['session']

    def eval( self, state, inference = False ) :
        _batchStates = [state] if state.ndim == 1 else state
        _qvalues = self._sess.run( self._opQhat_s, feed_dict = { self._tfStates : _batchStates } )

        ## set_trace()

        return _qvalues

    def train( self, states, actions, targets ) :

        ## set_trace()
        
        _, _loss = self._sess.run( [ self._opOptim, self._opLoss ],
                                   feed_dict = { self._tfStates : states,
                                                 self._tfActions : actions,
                                                 self._tfActionsIndices : np.arange( actions.shape[0] ),
                                                 self._tfQTargets : targets } )

        # grab loss for later statistics
        self._losses.append( _loss )

    def clone( self, other, tau = 1.0 ) :
        _srcWeights = self._nnetwork.get_weights()
        _dstWeights = other._nnetwork.get_weights()

        ## set_trace()

        _weights = []
        for i in range( len( _srcWeights ) ) :
            _weights.append( ( 1. - tau ) * _srcWeights[i] + ( tau ) * _dstWeights[i] )

        self._nnetwork.set_weights( _weights )

    def save( self, filename ) :
        self._nnetwork.save_weights( filename )

    def load( self, filename ) :
        self._nnetwork.load_weights( filename )


def BackendInitializer() :
    session = tf.InteractiveSession()
    session.run( tf.global_variables_initializer() )

    return { 'session' : session }

DqnModelBuilder = lambda name, config : DqnModelTensorflow( name, config )
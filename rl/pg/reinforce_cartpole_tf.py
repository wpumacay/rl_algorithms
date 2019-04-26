
import gym
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf

class Policy( object ) :

    def __init__( self, dimInput, dimOutput, dimHidden = 16, gamma = 0.9 ) :
        super( Policy, self ).__init__()

        # 
        self.m_gamma = gamma

        # input to the model (variable batch size)
        _input = tf.placeholder( tf.float32, name = 'input' )

        # declare the model parameters
        _w1 = tf.get_variable( 'w1',
                               dtype = tf.float32,
                               shape = ( dimHidden, dimInput ),
                               initializer = tf.random_normal_initializer() )
        _b1 = tf.get_variable( 'b1',
                               dtype = tf.float32,
                               shape = ( dimHidden, 1 ),
                               initializer = tf.random_normal_initializer() )
        _w2 = tf.get_variable( 'w2',
                               dtype = tf.float32,
                               shape = ( dimOutput, dimHidden ),
                               initializer = tf.random_normal_initializer() )
        _b2 = tf.get_variable( 'b2',
                               dtype = tf.float32,
                               shape = ( dimOutput, 1 ),
                               initializer = tf.random_normal_initializer() )

        # define the model architecture
        _a1 = tf.nn.relu( tf.add( tf.matmul( _w1, _input ), _b1 ) )
        _probs = tf.nn.softmax( tf.add( tf.matmul( _w2, _a1 ), _b2 ) )
        _logprobs = tf.log( _probs )
        _gradients = tf.gradients( _logprobs, [_w1, _b1, _w2, _b2] )

        # define advantage estimate placeholder
        _adv = tf.placeholder( tf.float32, name = 'advantage' )
        _loggradients = 

        # define pg loss
        _loss = tf.reduce_mean( tf.reduce_sum( -_logprobs * _adv ) )

        # define an optimizer to use
        _optim = tf.train.AdamOptimizer( learning_rate = 1e-2, name = "optim" )
        _trainOp = _optim.apply_gradients( _loss )

        # save definitions for later reference
        self.m_model = {}
        self.m_model['input']       = _input
        self.m_model['w1']          = _w1
        self.m_model['b1']          = _b1
        self.m_model['a1']          = _a1
        self.m_model['w2']          = _w2
        self.m_model['b2']          = _b2
        self.m_model['probs']       = _probs
        self.m_model['logprobs']    = _logprobs
        self.m_model['advantage']   = _adv
        self.m_model['loss']        = _loss

        self.m_actOp    = 
        self.m_trainOp  = _trainOp

        self.m_session = None

    def init( self, session ) :
        self.m_session = session
        self.m_session.run( tf.global_variables_initializer() )

    def act( self, state ) :
        if self.m_session is None :
            print( 'There must be a session for inference' )
            return None

        if self.m_session._closed :
            print( 'Current session is closed. Create a new one' )
            return None

        _args = { self.m_model['input'] : state }
        _prob = self.m_session.run( self.m_actOp, _args )
        _action = np.argmax( _prob )

        return _action

    def _computeReturns( self, rewards ) :
        # compute returns from state onwards
        _returns = []
        _G = 0
        for i in range( len( rewards ) - 1, -1, -1 ) :
            _G = rewards[i] + self.m_gamma * _G
            _returns.append( _G )

        return _returns

    def _computeAdvantages( self, returns ) :
        # simple advantage by removing mean ...
        # and normalizing with the stdev
        _mean  = np.mean( _returns )
        _stdev = np.std( _returns )

        return [ ( _ret - _mean ) / _stdev for _ret in returns ]

    def train( self, trajectory ) :
        # compute discounted returns
        trajectory['returns'] = self._computeReturns( trajectory['rewards'] )
        trajectory['advantages'] = self._computeAdvantages( trajectory['returns'] )
        






_env = gym.make( 'CartPole-v0' )
_env.seed( 0 )
print( 'S: ', _env.observation_space )
print( 'A: ', _env.action_space )

_policy = Policy( _env.observation_space.shape[0],
                  _env.action_space.n )

_state = _env.reset()
_snest = None

with tf.Session() as _session :
    _policy.init( _session )

    _trajectory = {}
    _trajectory['states'] = []
    _trajectory['actions'] = []
    _trajectory['rewards'] = []
    _trajectory['returns'] = None
    _trajectory['advantages'] = None

    for t in range( 2000 ) :
        _action = _policy.act( _state.reshape( ( 4, 1 ) ) )
        _snext, _reward, _done, _ = _env.step( _action )
        _env.render()

        _trajectory['states'].append( _state )
        _trajectory['actions'].append( _action )
        _trajectory['rewards'].append( _reward )
        _state = _snext

        if _done :
            break

    _trajectory['states'] = np.array( _trajectory['states'] )
    _policy.train( _trajectory )
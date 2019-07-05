
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import tensorflow as tf
import tensorflow.nn as nn

TRAIN = False# True
GAMMA = 1.0
LOG_WINDOW = 100

from IPython.core.debugger import set_trace

def create_policy_ops( input_shape, output_shape ) :
    _phInputs = tf.placeholder( tf.float64, shape = (None,) + input_shape, name = 'model_inputs' )
    # defining the weights and biases of our MLP backbone (reverse order shape to take batch-size into account)
    _fc1_w = tf.get_variable( 'model_fc1_w', shape = input_shape + (20,), dtype = tf.float64, trainable = True )
    _fc1_b = tf.get_variable( 'model_fc1_b', shape = (20,), dtype = tf.float64, trainable = True )
    _fc2_w = tf.get_variable( 'model_fc2_w', shape = (20,) + output_shape, dtype = tf.float64, trainable = True )
    _fc2_b = tf.get_variable( 'model_fc2_b', shape = output_shape, dtype = tf.float64, trainable = True )
    # constructing the model output
    _hidden1 = tf.nn.relu( tf.add( tf.matmul( _phInputs, _fc1_w ), _fc1_b ) )
    _opProbActions = tf.nn.softmax( tf.add( tf.matmul( _hidden1, _fc2_w ), _fc2_b ) )
    # placeholder to enter collected advantages (None,None) <> (#traj,#steps_in_traj)
    _phAdvantages = tf.placeholder( tf.float64, shape = (None,), name = 'policy_advantages' )
    # placeholder to enter the actions taken in the trajectory
    _phActionsTaken = tf.placeholder( tf.int64, shape = (None,), name = 'policy_action_indices' )
    # filter the actions taken in the trajectory
    _opActionsFilter = tf.stack( [ tf.range( tf.shape( _phActionsTaken )[0] ), tf.cast( _phActionsTaken, tf.int32 ) ], axis = 1 )
    _opProbActionsTaken = tf.gather_nd( _opProbActions, _opActionsFilter )
    # define the loss op and training op
    _opLoss = tf.reduce_sum( -tf.log( _opProbActionsTaken ) * _phAdvantages )
    _opOptim = tf.train.AdamOptimizer( learning_rate = 1e-3 ).minimize( _opLoss )

    return _phInputs, _opProbActions, _phAdvantages, _phActionsTaken, _opOptim


class Model:
    r"""Just a simple container of the model ops and placeholders"""

    def __init__( self, session, phInputs, phAdvantages, phActionsTaken, opProbActions, opOptim ) :
        self.session = session
        self.phInputs = phInputs
        self.phAdvantages = phAdvantages
        self.phActionsTaken = phActionsTaken
        self.opProbActions = opProbActions
        self.opOptim = opOptim


    def __call__( self, state ) :
        probActions = self.session.run( self.opProbActions, { self.phInputs : [state] } )[0]
        return probActions


    def train( self, states, advantages, actions ) :
        self.session.run( self.opOptim, { self.phInputs : states,
                                          self.phAdvantages : advantages,
                                          self.phActionsTaken : actions } )


def train( env, model, num_episodes = 2000 ) :
    env.seed( 0 )
    np.random.seed( 0 )

    scores_deque = deque( maxlen = LOG_WINDOW )
    mean_score = 0.
    best_score = -np.inf

    saver = tf.train.Saver()

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training> ' )

    for iepisode in progressbar :

        _s = env.reset()
        states = []
        actions = []
        rewards = []
        returns = []

        for _ in range( 1000 ) :
            # store state st for later training
            states.append( _s )
            # grab probabilities from model
            _probs = model( _s )
            # sample an action from the distribution
            _a = np.random.choice( env.action_space.n, p = _probs )
            # take the step in the environment and save transition
            _s, _r, _done, _ = env.step( _a )
            rewards.append( _r )
            actions.append( _a )

            if _done :
                break

        # update some info for logging
        episode_score = sum( rewards )
        best_score = max( best_score, episode_score )
        scores_deque.append( episode_score )

        # compute returns for each timestep (reverse order calculation)
        Gt = 0.
        for _r in reversed( rewards ) :
            Gt = _r + GAMMA * Gt
            returns.append( Gt )

        # reverse returns back to original ordering
        returns.reverse()
        # compute baseline(mean)
        g_mean  = np.mean( returns )
        advantages = [ ( gt - g_mean ) for gt in returns ]

        # train the model with the data so far
        model.train( states, advantages, actions )

        if iepisode >= LOG_WINDOW :
            mean_score = np.mean( scores_deque )
            message = 'Training> best: %.2f - mean: %.2f'
            progressbar.set_description( message % ( best_score, mean_score ) )
            progressbar.refresh()
        else :
            message = 'Training> best: %.2f'
            progressbar.set_description( message % ( best_score ) )
            progressbar.refresh()

    saver.save( model.session, './saved/tensorflow/vpg_cartpole.ckpt', write_meta_graph = False )


def test( env, model, num_episodes = 10 ) :
    saver = tf.train.Saver()
    saver.restore( model.session, './saved/tensorflow/vpg_cartpole.ckpt' )

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _s = env.reset()

        while not _done :
            _a = np.argmax( model( _s ) )
            _s, _r, _done, _ = env.step( _a )
            env.render()


if __name__ == '__main__' :
    with tf.Session() as session :
        env = gym.make( 'CartPole-v0' )

        phInputs, opProbActions, phAdvantages, phActionsTaken, opOptim = create_policy_ops( env.observation_space.shape,
                                                                                            (env.action_space.n,) )
        model = Model( session, phInputs, phAdvantages, phActionsTaken, opProbActions, opOptim )

        session.run( tf.global_variables_initializer() )

        if TRAIN :
            train( env, model )
        else :
            test( env, model )


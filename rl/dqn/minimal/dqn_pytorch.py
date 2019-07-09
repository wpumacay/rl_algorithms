
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as opt

from IPython.core.debugger import set_trace

TRAIN                   = False      # whether or not to train our agent
GAMMA                   = 1.0       # discount factor applied to the rewards
TAU                     = 0.001     # soft update factor used for target-network updates
EPS_DECAY_FACTOR        = 0.9985    # discount factory applied every episode
REPLAY_BUFFER_SIZE      = 100000    # size of the replay memory
LEARNING_RATE           = 0.0005    # learning rate used for action network
BATCH_SIZE              = 32        # batch size of data to grab for learning
TRAIN_FREQUENCY_STEPS   = 4         # learn every 4 steps (if there is data)
LOG_WINDOW              = 100       # size of the smoothing window and logging window
TRAINING_EPISODES       = 2000      # number of training episodes

DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

class QNetwork( nn.Module ) :

    def __init__( self, 
                  inputShape = (4,), 
                  outputShape = (2,),
                  layers_units = (128, 64, 16) ) :
        super( QNetwork, self ).__init__()

        assert len( layers_units ) == 3, 'ERROR> there should be 3 layers in this network'

        self._fc1 = nn.Linear( inputShape[0], layers_units[0] )
        self._fc2 = nn.Linear( layers_units[0], layers_units[1] )
        self._fc3 = nn.Linear( layers_units[1], layers_units[2] )
        self._fc4 = nn.Linear( layers_units[2], outputShape[0] )


    def forward( self, x ) :
        x = F.relu( self._fc1( x ) )
        x = F.relu( self._fc2( x ) )
        x = F.relu( self._fc3( x ) )
        x = self._fc4( x )

        return x


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class ReplayBuffer( object ) :

    def __init__( self, bufferSize ) :
        super( ReplayBuffer, self ).__init__()

        self._memory = deque( maxlen = bufferSize )


    def store( self, transition ) :
        self._memory.append( transition )


    def sample( self, batchSize ) :
        _batch = random.sample( self._memory, batchSize )

        _states = torch.tensor( [ _transition[0] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _actions = torch.tensor( [ _transition[1] for _transition in _batch ], dtype = torch.long ).to( DEVICE )
        _rewards = torch.tensor( [ _transition[2] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _statesNext = torch.tensor( [ _transition[3] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _dones = torch.tensor( [ _transition[4] for _transition in _batch ], dtype = torch.float ).to( DEVICE )

        return _states, _actions, _rewards, _statesNext, _dones


    def __len__( self ) :
        return len( self._memory )


def train( env, num_episodes = 2000 ) :

    scores = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    _qActionNetwork = QNetwork( env.observation_space.shape,
                                (env.action_space.n,) )
    _qActionNetwork.to( DEVICE )

    _qTargetNetwork = QNetwork( env.observation_space.shape,
                                (env.action_space.n,) )
    _qTargetNetwork.to( DEVICE )

    _qTargetNetwork.copy( _qActionNetwork )

    _rbuffer = ReplayBuffer( REPLAY_BUFFER_SIZE )

    epsilon = 1.0
    istep = 0

    optimizer = opt.Adam( _qActionNetwork.parameters(), lr = LEARNING_RATE )

    for iepisode in progressbar :
        _s = env.reset()
        _score = 0.
        _done = False

        while not _done :
            _qActionNetwork.eval()
            # eps-greedy strategy
            if np.random.rand() < epsilon :
                _a = np.random.randint( env.action_space.n )
            else :
                with torch.no_grad() :
                    _a = torch.argmax( _qActionNetwork( torch.from_numpy( _s ).float().to( DEVICE ) ) ).item()

            # apply action in the environment
            _snext, _r, _done, _ = env.step( _a )
            # store in the replay buffer
            _rbuffer.store( ( _s, _a, _r, _snext, _done ) )
            # book keeping for next iteration
            _s = _snext
            _score += _r
            istep += 1

            if len( _rbuffer ) > BATCH_SIZE and istep % TRAIN_FREQUENCY_STEPS == 0 :
                _qActionNetwork.train()
                # grab a sample from replay memory to train on
                _states, _actions, _rewards, _statesNext, _dones = _rbuffer.sample( BATCH_SIZE )
                # compute estimates of the q-values
                _qvalues = torch.gather( _qActionNetwork( _states ), 1, _actions.unsqueeze( 1 ).to( DEVICE ) )
                # compute targets to fit
                with torch.no_grad() :
                    _qtargets = (_rewards + ( 1. - _dones ) * GAMMA * torch.max( _qTargetNetwork( _statesNext ), 1 )[0]).unsqueeze( 1 )

                # reset the gradients buffer
                optimizer.zero_grad()

                # train on these targets
                loss_fcn = nn.MSELoss()
                ## set_trace()
                loss = loss_fcn( _qvalues, _qtargets )
                loss.backward()
                optimizer.step()

                # update target network parameters softly
                _qTargetNetwork.copy( _qActionNetwork, TAU )

        # update some info for logging
        bestScore = max( bestScore, _score )
        scoresWindow.append( _score )

        # update epsilon schedule
        epsilon = max( 0.01, epsilon * EPS_DECAY_FACTOR )

        if iepisode >= LOG_WINDOW :
            avgScore = np.mean( scoresWindow )
            message = 'Training> best: %.2f - mean: %.2f - current: %.2f - epsilon: %.2f'
            progressbar.set_description( message % ( bestScore, avgScore, _score, epsilon ) )
            progressbar.refresh()
        else :
            message = 'Training> best: %.2f - current : %.2f - epsilon: %.2f'
            progressbar.set_description( message % ( bestScore, _score, epsilon ) )
            progressbar.refresh()

    torch.save( _qActionNetwork.state_dict(), './saved/pytorch/dqn_cartpole.pth' )


def test( env, num_episodes = 10 ) :

    _qActionNetwork = QNetwork( env.observation_space.shape,
                                (env.action_space.n,) )
    _qActionNetwork.load_state_dict( torch.load( './saved/pytorch/dqn_cartpole.pth' ) )
    _qActionNetwork.eval()

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _s = env.reset()

        while not _done :
            _a = torch.argmax( _qActionNetwork( torch.from_numpy( _s ).float() ) ).item()
            _s, _r, _done, _ = env.step( _a )
            env.render()


if __name__ == '__main__' :

    env = gym.make( 'CartPole-v0' )
    env.seed( 0 )
    random.seed( 0 )
    np.random.seed( 0 )

    if TRAIN :
        train( env, TRAINING_EPISODES )
    else :
        test( env )

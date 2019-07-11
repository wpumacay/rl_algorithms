
import gym
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import torch
from torch import nn
from torch.functional import F
from torch import optim as opt

from tensorboardX import SummaryWriter

from IPython.core.debugger import set_trace

TRAIN                   = True      # whether or not to train our agent
GAMMA                   = 0.99      # discount factor applied to the rewards
TAU                     = 0.001     # soft update factor used for target-network updates
REPLAY_BUFFER_SIZE      = 1000000   # size of the replay memory
LEARNING_RATE_ACTOR     = 0.0001    # learning rate used for actor network
LEARNING_RATE_CRITIC    = 0.0003    # learning rate used for the critic network
BATCH_SIZE              = 128       # batch size of data to grab for learning
TRAIN_FREQUENCY_STEPS   = 1         # learn every 4 steps (if there is data)
LOG_WINDOW              = 100       # size of the smoothing window and logging window
TRAINING_EPISODES       = 2000      # number of training episodes
MAX_STEPS_IN_EPISODE    = 700       # maximum number of steps in an episode
SEED                    = 10        # random seed to be used
WEIGHT_DECAY            = 0.0001    # L2 weight decay used in Adam

DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

# layer weights initializer, from udacity-deeprl example
def hidden_init( layer ) :
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt( fan_in )
    return ( -lim, lim )


class PiNetwork( nn.Module ) :
    r"""A simple deterministic policy network class to be used for the actor

    Args:
        inputShape (tuple): shape of the state space
        outputShape (tuple): shape of the action space

    """
    def __init__( self, inputShape, outputShape ) :
        super( PiNetwork, self ).__init__()

        self.seed = torch.manual_seed( SEED )
        self.fc1 = nn.Linear( inputShape[0], 256 )
        self.fc2 = nn.Linear( 256, outputShape[0] )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *hidden_init( self.fc1 ) )
        self.fc2.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, state ) :
        r"""Forward pass for this deterministic policy, used for the max Q evaluation

        Args:
            state (torch.tensor): state used to decide the action

        """
        x = F.relu( self.fc1( state ) )
        x = F.tanh( self.fc2( x ) )

        return x


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class Qnetwork( nn.Module ) :
    r"""A simple Q-network class to be used for the critic

    """
    def __init__( self, inputShape, outputShape ) :
        super( Qnetwork, self ).__init__()

        self.seed = torch.manual_seed( SEED )
        self.fc1 = nn.Linear( inputShape[0], 256 )
        self.fc2 = nn.Linear( 256 + outputShape[0], 256 )
        self.fc3 = nn.Linear( 256, 128 )
        self.fc4 = nn.Linear( 128, 1 )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *hidden_init( self.fc1 ) )
        self.fc2.weight.data.uniform_( *hidden_init( self.fc2 ) )
        self.fc3.weight.data.uniform_( *hidden_init( self.fc3 ) )
        self.fc4.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, state, action ) :
        r"""Forward pass for this critic at a given (s,a) pair

        Args:
            state (torch.tensor): state of the pair to be evaluated
            action (torch.tensor): action of the pair to be evaluated

        """
        xs = F.leaky_relu( self.fc1( state ) )
        xaug = torch.cat( [xs, action], dim = 1 )
        x = F.leaky_relu( self.fc2( xaug ) )
        x = F.leaky_relu( self.fc3( x ) )
        x = self.fc4( x )

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
        _actions = torch.tensor( [ _transition[1] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _rewards = torch.tensor( [ _transition[2] for _transition in _batch ], dtype = torch.float ).unsqueeze( 1 ).to( DEVICE )
        _statesNext = torch.tensor( [ _transition[3] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _dones = torch.tensor( [ _transition[4] for _transition in _batch ], dtype = torch.float ).unsqueeze( 1 ).to( DEVICE )

        return _states, _actions, _rewards, _statesNext, _dones


    def __len__( self ) :
        return len( self._memory )


class OUNoise( object ) :

    def __init__( self, size, mu = 0., theta = 0.15, sigma = 0.2 ) :
        super( OUNoise, self ).__init__()

        self._mu = mu * np.ones( size )
        self._theta = theta
        self._sigma = sigma
        self._state = self._mu.copy()


    def reset( self ) :
        self._state = self._mu.copy()


    def sample( self ) :
        x = self._state
        dx = self._theta * ( self._mu - x ) + self._sigma * np.random.rand( *self._mu.shape )
        self._state = x + dx

        return self._state.copy()


class OUNoise2:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(SEED)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def train( env, num_episodes = 2000 ) :
    _actorNetLocal = PiNetwork( env.observation_space.shape,
                                env.action_space.shape ).to( DEVICE )
    _actorNetTarget = PiNetwork( env.observation_space.shape,
                                 env.action_space.shape ).to( DEVICE )
    # _actorNetTarget.copy( _actorNetLocal )

    _criticNetLocal = Qnetwork( env.observation_space.shape,
                                env.action_space.shape ).to( DEVICE )
    _criticNetTarget = Qnetwork( env.observation_space.shape,
                                 env.action_space.shape ).to( DEVICE )
    # _criticNetTarget.copy( _criticNetLocal )

    _rbuffer = ReplayBuffer( REPLAY_BUFFER_SIZE )
    ## _noise = OUNoise( env.action_space.shape )
    _noise = OUNoise2( env.action_space.shape )

    _optimActor = opt.Adam( _actorNetLocal.parameters(), lr = LEARNING_RATE_ACTOR )
    _optimCritic = opt.Adam( _criticNetLocal.parameters(), lr = LEARNING_RATE_CRITIC, weight_decay = WEIGHT_DECAY )

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( 'summary_bipedal' )
    istep = 0

    for iepisode in progressbar :

        _s = env.reset()
        _noise.reset()
        _score = 0.

        for _ in range( MAX_STEPS_IN_EPISODE ) :
            # choose an action using the actor network and a noise process
            _actorNetLocal.eval()
            with torch.no_grad() :
                _a = _actorNetLocal( torch.from_numpy( _s ).float().to( DEVICE ) ).cpu().data.numpy()
            _actorNetLocal.train()
            # add noise and clip accordingly
            _a += _noise.sample()
            _a = np.clip( _a, -1., 1. )

            # take action in the environment and grab bounty
            _snext, _r, _done, _ = env.step( _a )
            _rbuffer.store( (_s, _a, _r, _snext, _done ) )

            if len( _rbuffer ) > BATCH_SIZE and istep % TRAIN_FREQUENCY_STEPS == 0 :
                ## set_trace()
                # grab a batch of data from the replay buffer
                _states, _actions, _rewards, _statesNext, _dones = _rbuffer.sample( BATCH_SIZE )
                # compute current q-values for the 'actions' taken at 'states' using critic
                _qvalues = _criticNetLocal( _states, _actions )
                # compute target q-values using both target actor and critic
                with torch.no_grad() :
                    _actionsNext = _actorNetTarget( _statesNext )
                    _qvaluesTarget = _rewards + ( 1. - _dones ) * GAMMA * _criticNetTarget( _statesNext, _actionsNext )

                # compute loss for the critic
                _optimCritic.zero_grad()
                _lossCritic = F.mse_loss( _qvalues, _qvaluesTarget )
                _lossCritic.backward()
                _optimCritic.step()

                # compute loss for the actor, from the objective to "maximize":
                #
                # dJ / dtheta = E [ dQ / du * du / dtheta ]
                #
                # where:
                #   * theta: weights of the actor
                #   * dQ / du : gradient of Q w.r.t. u (actions taken)
                #   * du / dtheta : gradient of the Actor's weights

                _optimActor.zero_grad()
                # compute actions taken in these states by the current state of the actor
                _actionsPred = _actorNetLocal( _states )
                # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
                _lossActor = -_criticNetLocal( _states, _actionsPred ).mean()
                _lossActor.backward()
                _optimActor.step()

                # update target networks
                _actorNetTarget.copy( _actorNetLocal, TAU )
                _criticNetTarget.copy( _criticNetLocal, TAU )

            # book keeping for next iteration
            _s = _snext
            _score += _r
            istep += 1

            if _done :
                break

        # update some info for logging
        bestScore = max( bestScore, _score )
        scoresWindow.append( _score )

        if iepisode >= LOG_WINDOW :
            avgScore = np.mean( scoresWindow )
            scoresAvgs.append( avgScore )
            message = 'Training> best: %.2f - mean: %.2f - current: %.2f'
            progressbar.set_description( message % ( bestScore, avgScore, _score ) )
            progressbar.refresh()
        else :
            message = 'Training> best: %.2f - current : %.2f'
            progressbar.set_description( message % ( bestScore, _score ) )
            progressbar.refresh()

        writer.add_scalar( 'score', _score, iepisode )
        writer.add_scalar( 'avg_score', np.mean( scoresWindow ), iepisode )

    torch.save( _actorNetLocal.state_dict(), './saved/pytorch/ddpg_actor_bipedal.pth' )
    torch.save( _criticNetLocal.state_dict(), './saved/pytorch/ddpg_critic_bipedal.pth' )


def test( env, num_episodes = 10 ) :
    _actorNet = PiNetwork( env.observation_space.shape,
                           env.action_space.shape )

    _actorNet.load_state_dict( torch.load( './saved/pytorch/ddpg_actor_bipedal.pth' ) )
    _actorNet.eval()

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _s = env.reset()

        while not _done :
            _a = _actorNet( torch.from_numpy( _s ).float() ).cpu().data.numpy()
            _s, _r, _done, _ = env.step( _a )
            env.render()


if __name__ == '__main__' :
    env = gym.make( 'BipedalWalker-v2' )

    env.seed( SEED )
    random.seed( SEED )
    np.random.seed( SEED )
    torch.manual_seed( SEED )

    if TRAIN :
        train( env, TRAINING_EPISODES )
    else :
        test( env )
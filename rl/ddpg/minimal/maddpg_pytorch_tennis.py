
import os
import gym
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import torch
from torch import nn
from torch.functional import F
from torch import optim as opt

from rl.envs.mlagents_env_wrapper import createMultiEnvWrapper

from tensorboardX import SummaryWriter

from IPython.core.debugger import set_trace

NUM_AGENTS              = 2         # number of agents in the multiagent env. setup
TRAIN                   = True      # whether or not to train our agent
GAMMA                   = 0.99      # discount factor applied to the rewards
TAU                     = 0.001     # soft update factor used for target-network updates
REPLAY_BUFFER_SIZE      = 1000000   # size of the replay memory
LEARNING_RATE_ACTOR     = 0.001     # learning rate used for actor network
LEARNING_RATE_CRITIC    = 0.001     # learning rate used for the critic network
BATCH_SIZE              = 256       # batch size of data to grab for learning
TRAIN_FREQUENCY_STEPS   = 4         # learn every 10 steps (if there is data)
TRAIN_NUM_UPDATES       = 2         # number of updates to do when doing a learning
LOG_WINDOW              = 100       # size of the smoothing window and logging window
TRAINING_EPISODES       = 50000     # number of training episodes
MAX_STEPS_IN_EPISODE    = 3000      # maximum number of steps in an episode
SEED                    = 200         # random seed to be used
EPSILON_SCHEDULE        = 'linear'  # type of shedule 
EPSILON_DECAY_FACTOR    = 0.999     # decay factor for e-greedy geometric schedule
EPSILON_DECAY_LINEAR    = 2e-5      # decay factor for e-greedy linear schedule
TRAINING_STARTING_STEP  = int(5e4)  # step index at which training should start
TRAINING_SESSION_ID     = 'sess_0'  # name of the training session

DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

def hidden_init( layer ) :
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt( fan_in / 2 )
    return ( -lim, lim )


class PiNetwork( nn.Module ) :
    r"""A simple deterministic policy network class to be used for the actor

    Args:
        observationShape (tuple): shape of the observations given to the network
        actionShape (tuple): shape of the actions to be computed by the network

    """
    def __init__( self, observationShape, actionShape ) :
        super( PiNetwork, self ).__init__()

        self.seed = torch.manual_seed( SEED )
        self.bn0 = nn.BatchNorm1d( observationShape[0] )
        self.fc1 = nn.Linear( observationShape[0], 256 )
        self.bn1 = nn.BatchNorm1d( 256 )
        self.fc2 = nn.Linear( 256, 128 )
        self.bn2 = nn.BatchNorm1d( 128 )
        self.fc3 = nn.Linear( 128, actionShape[0] )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *hidden_init( self.fc1 ) )
        self.fc2.weight.data.uniform_( *hidden_init( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, observation ) :
        r"""Forward pass for this deterministic policy, used for the max Q evaluation

        Args:
            observation (torch.tensor): observation used to decide the action

        """
        x = self.bn0( observation )
        x = F.relu( self.bn1( self.fc1( x ) ) )
        x = F.relu( self.bn2( self.fc2( x ) ) )
        x = F.tanh( self.fc3( x ) )

        return x


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class Qnetwork( nn.Module ) :
    r"""A simple Q-network class to be used for the centralized critics

    Args:
        jointObservationShape (tuple): shape of the augmented state representation [o1,o2,...on]
        jointActionShape (tuple): shape of the augmented action representation [a1,a2,...,an]

    """
    def __init__( self, jointObservationShape, jointActionShape ) :
        super( Qnetwork, self ).__init__()

        self.seed = torch.manual_seed( SEED )

        self.bn0 = nn.BatchNorm1d( jointObservationShape[0] )
        self.fc1 = nn.Linear( jointObservationShape[0], 128 )
        self.fc2 = nn.Linear( 128 + jointActionShape[0], 128 )
        self.fc3 = nn.Linear( 128, 1 )
        self._init()


    def _init( self ) :
        self.fc1.weight.data.uniform_( *hidden_init( self.fc1 ) )
        self.fc2.weight.data.uniform_( *hidden_init( self.fc2 ) )
        self.fc3.weight.data.uniform_( -3e-3, 3e-3 )


    def forward( self, jointObservation, jointAction ) :
        r"""Forward pass for this critic at a given (x=[o1,...,an],aa=[a1...an]) pair

        Args:
            jointObservation (torch.tensor): augmented observation [o1,o2,...,on]
            jointAction (torch.tensor): augmented action [a1,a2,...,an]

        """
        _h = self.bn0( jointObservation )
        _h = F.relu( self.fc1( _h ) )
        _h = torch.cat( [_h, jointAction], dim = 1 )
        _h = F.relu( self.fc2( _h ) )
        _h = self.fc3( _h )

        return _h


    def copy( self, other, tau = 1.0 ) :
        for paramsSelf, paramsOther in zip( self.parameters(), other.parameters() ) :
            paramsSelf.data.copy_( ( 1. - tau ) * paramsSelf.data + tau * paramsOther.data )


class ReplayBuffer( object ) :
    r"""Replay buffer class used to train centralized critics.

    This replay buffer is the same as our old friend the replay-buffer from
    the vanilla dqn for a single agent, with some slight variations as the 
    tuples stored now consist in some cases in augmentations of the observations
    and action spaces:

    ([o1,...,on],[a1,...,an],[r1,...,rn],[o1',...,on'],[d1,...,dn])
          x                                    x'

    The usage depends on the network that will consume this data in its forward
    pass, which could be either a decentralized actor or a centralized critic.

    For a decentralized actor:

        u    ( oi ) requires the local observation for that actor
         theta-i

    For a centralized critic:

        Q     ( [o1,...,on], [a1,...,an] ) requires both the augmented observation
         phi-i   ----------  -----------   and the joint action from the actors
                     |            |
                     x        joint-action

    So, to make things simpler, as the environment is already returning packed
    numpy ndarrays with first dimension equal to the num-agents, we will store
    these as when sampling a minibatch we will actually returned an even more
    packed version, which would include a batch dimension on top of the over
    dimensions (n-agents,variable-shape), so we would have something like:
    
    e.g. storing:

        store( ( [obs1(33,),obs2(33,)], [a1(4,),a2(4,)], ... ) )
                 --------------------   ---------------
                    ndarray(2,33)         ndarray(2,4)

    e.g. sampling:
        batch -> ( batchObservations, batchActions, ... )
                   -----------------  ------------
                    tensor(128,2,33)   tensor(128,2,4)

    Args:
        bufferSize (int): max. number of experience tuples this buffer will hold
                          until it starts throwing away old experiences in a FIFO
                          way.
        numAgents (int): number of agents used during learning (for sanity-checks)

    """

    def __init__( self, bufferSize, numAgents ) :
        super( ReplayBuffer, self ).__init__()

        self._memory = deque( maxlen = bufferSize )
        self._numAgents = numAgents


    def store( self, transition ) :
        r"""Stores a transition tuple in memory

        The transition tuples to be stored must come in the form:

        ( [o1,...,on], [a1,...,an], [r1,...,rn], [o1',...,on'], [done1,...,donen] )

        Args:
            transition (tuple): a transition tuple to be stored in memory

        """
        # sanity-check: ensure first dimension of each transition component has the right size
        assert len( transition[0] ) == self._numAgents, 'ERROR> group observation size mismatch'
        assert len( transition[1] ) == self._numAgents, 'ERROR> group actions size mismatch'
        assert len( transition[2] ) == self._numAgents, 'ERROR> group rewards size mismatch'
        assert len( transition[3] ) == self._numAgents, 'ERROR> group next observations size mismatch'
        assert len( transition[4] ) == self._numAgents, 'ERROR> group dones size mismatch'

        self._memory.append( transition )


    def sample( self, batchSize ) :
        _batch = random.sample( self._memory, batchSize )

        _observations       = torch.tensor( [ _transition[0] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _actions            = torch.tensor( [ _transition[1] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _rewards            = torch.tensor( [ _transition[2] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )
        _observationsNext   = torch.tensor( [ _transition[3] for _transition in _batch ], dtype = torch.float ).to( DEVICE )
        _dones              = torch.tensor( [ _transition[4] for _transition in _batch ], dtype = torch.float ).unsqueeze( 2 ).to( DEVICE )

        return _observations, _actions, _rewards, _observationsNext, _dones


    def __len__( self ) :
        return len( self._memory )


class OUNoise( object ) :

    def __init__( self, size, mu = 0., theta = 0.15, sigma = 0.05 ) :
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
    actorsNetsLocal = [ PiNetwork( env.observation_space.shape,
                                   env.action_space.shape ) for _ in range( NUM_AGENTS ) ]
    actorsNetsTarget = [ PiNetwork( env.observation_space.shape,
                                    env.action_space.shape ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( actorsNetsLocal, actorsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    criticsNetsLocal = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                   (NUM_AGENTS * env.action_space.shape[0],) ) for _ in range( NUM_AGENTS ) ]
    criticsNetsTarget = [ Qnetwork( (NUM_AGENTS * env.observation_space.shape[0],),
                                    (NUM_AGENTS * env.action_space.shape[0],) ) for _ in range( NUM_AGENTS ) ]
    for _netLocal, _netTarget in zip( criticsNetsLocal, criticsNetsTarget ) :
        _netTarget.copy( _netLocal )
        _netLocal.to( DEVICE )
        _netTarget.to( DEVICE )

    optimsActors = [ opt.Adam( _actorNet.parameters(), lr = LEARNING_RATE_ACTOR ) \
                        for _actorNet in actorsNetsLocal ]
    optimsCritics = [ opt.Adam( _criticNet.parameters(), lr = LEARNING_RATE_CRITIC ) \
                        for _criticNet in criticsNetsLocal ]

    rbuffer = ReplayBuffer( REPLAY_BUFFER_SIZE, NUM_AGENTS )
    noise = OUNoise2( env.action_space.shape )

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training>' )

    scoresAvgs = []
    scoresWindow = deque( maxlen = LOG_WINDOW )
    bestScore = -np.inf
    avgScore = -np.inf

    writer = SummaryWriter( 'summary_maddpg_' + TRAINING_SESSION_ID + '_reacher_bn' )
    istep = 0
    epsilon = 1.0

    for iepisode in progressbar :

        noise.reset()
        _oo = env.reset()
        _scoreAgents = np.zeros( NUM_AGENTS )

        for i in range( MAX_STEPS_IN_EPISODE ) :
            if istep < TRAINING_STARTING_STEP :
                _aa = np.clip( np.random.randn( *((NUM_AGENTS,) + env.action_space.shape) ), -1., 1. )
            else :
                for _actorNet in actorsNetsLocal :
                    _actorNet.eval()
                # choose an action for each agent using its own actor network
                with torch.no_grad() :
                    _aa = []
                    for iactor, _actorNet in enumerate( actorsNetsLocal ) :
                        _a = _actorNet( torch.from_numpy( _oo[iactor] ).unsqueeze( 0 ).float().to( DEVICE ) ).cpu().data.numpy().squeeze()
                        _aa.append( _a )
                    _aa = np.array( _aa )
                    _nn = np.array( [ epsilon * noise.sample() for _ in range( NUM_AGENTS ) ] ).reshape( _aa.shape )
                    _aa += _nn
                    _aa = np.clip( _aa, -1., 1. ) # actions are speed-factors (range (-1,1)) in both x and y
                for _actorNet in actorsNetsLocal :
                    _actorNet.train()

            # take action in the environment and grab bounty
            _oonext, _rr, _dd, _ = env.step( _aa )
            # store joint information (form (NAGENTS,) + MEASUREMENT-SHAPE)
            if i == MAX_STEPS_IN_EPISODE - 1 :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, np.ones_like( _dd ) ) )
            else :
                rbuffer.store( ( _oo, _aa, _rr, _oonext, _dd ) )

            if len( rbuffer ) > BATCH_SIZE and istep % TRAIN_FREQUENCY_STEPS == 0 and \
               istep >= TRAINING_STARTING_STEP :
                for _ in range( TRAIN_NUM_UPDATES ) :
                    # grab a batch of data from the replay buffer
                    _observations, _actions, _rewards, _observationsNext, _dones = rbuffer.sample( BATCH_SIZE )

                    # compute joint observations and actions to be passed ...
                    # to the critic, which basically consists of keep the ...
                    # batch dimension and vectorize everything else into one ...
                    # single dimension [o1,...,on] and [a1,...,an]
                    _batchJointObservations = _observations.reshape( _observations.shape[0], -1 )
                    _batchJointObservationsNext = _observationsNext.reshape( _observationsNext.shape[0], -1 )
                    _batchJointActions = _actions.reshape( _actions.shape[0], -1 )

                    # compute the joint next actions required for the centralized ...
                    # critics q-target computation
                    with torch.no_grad() :
                        _batchJointActionsNext = torch.stack( [ actorsNetsTarget[iactor]( _observationsNext[:,iactor,:] )  \
                                                                for iactor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsNext = _batchJointActionsNext.reshape( _batchJointActionsNext.shape[0], -1 )

                    for iactor in range( NUM_AGENTS ) :

                        #---------------------- TRAIN CRITICS  --------------------#

                        # extract local observations to be fed to the actors, ...
                        # as well as local rewards and dones to be used for local 
                        # q-targets computation using critics
                        _batchLocalObservations = _observations[:,iactor,:]
                        _batchLocalRewards = _rewards[:,iactor,:]
                        _batchLocalDones = _dones[:,iactor,:]
                        # compute current q-values for the joint-actions taken ...
                        # at joint-observations using the critic, as explained ...
                        # in the MADDPG algorithm:
                        #
                        # Q(x,a1,a2,...,an) -> Q( [o1,o2,...,on], [a1,a2,...,an] )
                        #                       phi-i
                        _qvalues = criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActions )
                        # compute target q-values using both decentralized ...
                        # target actor and centralized target critic for this ...
                        # current actor, as explained in the MADDPG algorithm:
                        #
                        # Q-targets  = r  + ( 1 - done ) * gamma * Q  ( [o1',...,on'], [a1',...,an'] )
                        #          i    i             i             phi-target-i
                        # 
                        # 
                        with torch.no_grad() :
                            _qvaluesTarget = _batchLocalRewards + ( 1. - _batchLocalDones ) \
                                                * GAMMA * criticsNetsTarget[iactor]( _batchJointObservationsNext, 
                                                                                     _batchJointActionsNext )
        
                        # compute loss for the critic
                        optimsCritics[iactor].zero_grad()
                        _lossCritic = F.mse_loss( _qvalues, _qvaluesTarget )
                        _lossCritic.backward()
                        torch.nn.utils.clip_grad_norm( criticsNetsLocal[iactor].parameters(), 1 )
                        optimsCritics[iactor].step()
    
                        #---------------------- TRAIN ACTORS  ---------------------#
    
                        # compute loss for the actor, from the objective to "maximize":
                        #
                        # dJ / dtheta = E [ dQ / du * du / dtheta ]
                        #
                        # where:
                        #   * theta: weights of the actor
                        #   * dQ / du : gradient of Q w.r.t. u (actions taken)
                        #   * du / dtheta : gradient of the Actor's weights
        
                        optimsActors[iactor].zero_grad()

                        # compute predicted actions for current local observations ...
                        # as we will need them for computing the gradients of the ...
                        # actor. Recall that these gradients depend on the gradients ...
                        # of its own related centralized critic, which need the joint ...
                        # actions to work. Keep with grads here as we have to build ...
                        # the computation graph with these operations

                        _batchJointActionsPred = torch.stack( [ actorsNetsLocal[indexActor]( _observations[:,indexActor,:] )  \
                                                                    for indexActor in range( NUM_AGENTS ) ], dim = 1 )
                        _batchJointActionsPred = _batchJointActionsPred.reshape( _batchJointActionsPred.shape[0], -1 )

                        # compose the critic over the actor outputs (sandwich), which effectively does g(f(x))
                        _lossActor = -criticsNetsLocal[iactor]( _batchJointObservations, _batchJointActionsPred ).mean()
                        _lossActor.backward()
                        optimsActors[iactor].step()
        
                        # update target networks
                        actorsNetsTarget[iactor].copy( actorsNetsLocal[iactor], TAU )
                        criticsNetsTarget[iactor].copy( criticsNetsLocal[iactor], TAU )
    
                    # update epsilon using schedule
                    if EPSILON_SCHEDULE == 'linear' :
                        epsilon = max( 0.1, epsilon - EPSILON_DECAY_LINEAR )
                    else :
                        epsilon = max( 0.1, epsilon * EPSILON_DECAY_FACTOR )

                for iactor in range( NUM_AGENTS ) :
                    torch.save( actorsNetsLocal[iactor].state_dict(), './saved/pytorch/maddpg_actor_reacher_' + str(iactor) + '_' + TRAINING_SESSION_ID + '.pth' )
                    torch.save( criticsNetsLocal[iactor].state_dict(), './saved/pytorch/maddpg_critic_reacher_' + str(iactor) + '_' + TRAINING_SESSION_ID + '.pth' )

            # book keeping for next iteration
            _oo = _oonext
            _scoreAgents += _rr
            istep += 1

            ## if np.mean( _rr ) > 0. :
            ##     set_trace()

            if _dd.any() :
                break

        # update some info for logging
        _score = np.max( _scoreAgents ) # score of the game is the max over both agents' scores
        bestScore = max( bestScore, _score ) # max game score so far
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
        writer.add_scalar( 'buffer_size', len( rbuffer ), iepisode )
        writer.add_scalar( 'epsilon', epsilon, iepisode )

    for iactor in range( NUM_AGENTS ) :
        torch.save( actorsNetsLocal[iactor].state_dict(), './saved/pytorch/maddpg_actor_reacher_' + str(iactor) + '_' + TRAINING_SESSION_ID + '.pth' )
        torch.save( criticsNetsLocal[iactor].state_dict(), './saved/pytorch/maddpg_critic_reacher_' + str(iactor) + '_' + TRAINING_SESSION_ID + '.pth' )


def test( env, num_episodes = 10 ) :
    actorsNets = [ PiNetwork( env.observation_space.shape,
                              env.action_space.shape ) for _ in range( NUM_AGENTS ) ]
    for iactor, _actorNet in enumerate( actorsNets ) :
        _actorNet.load_state_dict( torch.load( './saved/pytorch/maddpg_actor_reacher_' + str( iactor ) + '_' + TRAINING_SESSION_ID + '.pth' ) )
        _actorNet.eval()

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _oo = env.reset()

        while not _done :
            # compute actions for each actor
            _aa = []
            for iactor, _actorNet in enumerate( actorsNets ) :
                _a = _actorNet( torch.from_numpy( _oo[iactor] ).unsqueeze( 0 ).float() ).data.numpy().squeeze()
                _aa.append( _a )
            _aa = np.array( _aa )

            _oo, _rr, _dd, _ = env.step( _aa )
            env.render()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument( 'mode', help='mode to run the script (train|test)', type=str, choices=['train','test'], default='train' )
    parser.add_argument( '--sessionId', help='unique identifier of this training run', type=str, default='session_default' )
    parser.add_argument( '--hp_replay_buffer_size', help='size of the replay buffer to be used', type=int, default=REPLAY_BUFFER_SIZE )
    parser.add_argument( '--hp_batch_size', help='batch size for updates on both the actor and critic', type=int, default=BATCH_SIZE )
    parser.add_argument( '--hp_lrate_actor', help='learning rate used for the actor', type=float, default=LEARNING_RATE_ACTOR )
    parser.add_argument( '--hp_lrate_critic', help='learning rate used for the critic', type=float, default=LEARNING_RATE_CRITIC )
    parser.add_argument( '--hp_tau', help='soft update parameter (polyak averaging)', type=float, default=TAU )
    parser.add_argument( '--hp_train_update_freq', help='how often to do a learning step', type=int, default=TRAIN_FREQUENCY_STEPS )
    parser.add_argument( '--hp_train_num_updates', help='how many updates to do per learning step', type=int, default=TRAIN_NUM_UPDATES )

    args = parser.parse_args()

    TRAIN                   = ( args.mode.lower() == 'train' )
    TRAINING_SESSION_ID     = args.sessionId
    REPLAY_BUFFER_SIZE      = args.hp_replay_buffer_size
    BATCH_SIZE              = args.hp_batch_size
    LEARNING_RATE_ACTOR     = args.hp_lrate_actor
    LEARNING_RATE_CRITIC    = args.hp_lrate_critic
    TAU                     = args.hp_tau
    TRAIN_FREQUENCY_STEPS   = args.hp_train_update_freq
    TRAIN_NUM_UPDATES       = args.hp_train_num_updates

    print( '#############################################################' )
    print( '#                                                           #' )
    print( '#            Environment and agent setup                    #' )
    print( '#                                                           #' )
    print( '#############################################################' )
    print( 'Mode                    : ', args.mode.lower() )
    print( 'SessionId               : ', args.sessionId )
    print( 'Replay buffer size      : ', args.hp_replay_buffer_size )
    print( 'Batch size              : ', args.hp_batch_size )
    print( 'Learning-rate actor     : ', args.hp_lrate_actor )
    print( 'Learning-rate critic    : ', args.hp_lrate_critic )
    print( 'Tau                     : ', args.hp_tau )
    print( 'Train update freq       : ', args.hp_train_update_freq )
    print( 'Train num updates       : ', args.hp_train_num_updates )
    print( '#############################################################' )

    # create the environment
    execPath = os.path.join( os.getcwd(), '../../../envs/Tennis_Linux/Tennis.x86_64' )
    env = createMultiEnvWrapper( execPath, 
                                 numAgents = 2, 
                                 mode = 'training' if TRAIN else 'test',
                                 workerID = 1, seed = SEED )

    env.seed( SEED )
    random.seed( SEED )
    np.random.seed( SEED )
    torch.manual_seed( SEED )

    if TRAIN :
        train( env, TRAINING_EPISODES )
    else :
        test( env )

import gym
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
torch.manual_seed( 0 )
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical

plt.ion()

DEBUG_ANALYZE_GRAPH = True
# BATCH_SIZE = 10

# create the environment
_env = gym.make( 'CartPole-v0' )
## _env = gym.make( 'MountainCar-v0' )
_env.seed( 0 )
print( 'S: ', _env.observation_space )
print( 'A: ', _env.action_space )

# initialize pytorch-cuda device
_device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

# define architecture of the policy network
# MLP with a single hidden layer

class Policy( nn.Module ) :

    def __init__( self, dimInput = 4, dimOutput = 2, dimHidden = 20 ) :
        super( Policy, self ).__init__()

        self.fc1 = nn.Linear( dimInput, dimHidden )
        self.fc2 = nn.Linear( dimHidden, dimOutput )

    def forward( self, s ) :
        s = func.relu( self.fc1( s ) )
        s = func.softmax( self.fc2( s ), dim = 1 )
        return s

    def act( self, s ) :
        # transform state to tensor and send to device
        s = torch.from_numpy( s ).float().unsqueeze( 0 ).to( _device )
        # compute the probabilities p = pi(a|s) for all actions using the policy
        p = self.forward( s ).cpu()
        # grab the action using those probabilities (stochastic)
        pdist = Categorical( p )
        a = pdist.sample()

        # return the action (to execute) and the log probability
        return a.item(), pdist.log_prob( a )

_policy = Policy( _env.observation_space.shape[0], 
                  _env.action_space.n ).to( _device )
_optimizer = optim.Adam( _policy.parameters(), lr = 1e-2 )

# training with REINFORCE algorithm
def reinforce( nEpisodes = 1000, 
               nMaxStepsEpisode = 1000,
               gamma = 1.0,
               printEvery = 100 ) :
    
    # for 100 most recent
    _scoresDeque = deque( maxlen = 100 )
    # for all history of scores
    _scoresBuffer = []

    for iEpisode in tqdm( range( nEpisodes ) ) :
        _logProbsBuffer = []
        _rewardsBuffer = []
        _state = _env.reset()

        # collect m=1 episode
        for _ in range( nMaxStepsEpisode ) :
            # use policy network to get action and log probability
            _action, _logProb = _policy.act( _state )
            _logProbsBuffer.append( _logProb )
            # take a step in the simulation
            _state, _reward, _done, _ = _env.step( _action )
            _rewardsBuffer.append( _reward )
            if _done :
                break

        _scoresDeque.append( sum( _rewardsBuffer ) )
        _scoresBuffer.append( sum( _rewardsBuffer ) )

        # _npRewards = np.array( _rewardsBuffer )
        # _npRewards = ( _npRewards - np.mean( _npRewards ) ) / ( np.std( _npRewards ) + 0.0000001 )
        _discounts = [ gamma ** i for i in range( len( _rewardsBuffer ) + 1 ) ]
        _R = sum( [ a * b for a, b in zip( _discounts, _rewardsBuffer ) ] )

        _policyLoss = []
        for _logProb in _logProbsBuffer :
            # change sign to use SGD (not ascent)
            _policyLoss.append( -_logProb * _R )

        # concatenate all losses into tensor and compute the sum of this tensor
        # (garbage collector, I choose you!) just reassign the ref. variable and
        # let gc go its job. The loss is of type tensor (check type<>) so can
        # access the gradient from autograd with backward
        _policyLoss = torch.cat( _policyLoss ).sum()

        # do backprop
        _optimizer.zero_grad()
        _policyLoss.backward()
        _optimizer.step()

        _meanScore = np.mean( _scoresDeque )
        if iEpisode % printEvery == 0 :
            print( 'Episode {}\tAverage Score: {:.2f}'.format( iEpisode, _meanScore ) )
        if _meanScore >= 195.0 :
            print( 'Environment solved in {:d} episodes\tAverage score: {:.2f}'.format( iEpisode, _meanScore ) )
            break

    return _scoresBuffer

_scores = reinforce()

# plot the training results
_fig = plt.figure()
_ax = _fig.add_subplot( 111 )
plt.plot( np.arange( 1, len( _scores ) + 1 ), _scores )
plt.ylabel( 'Score' )
plt.xlabel( 'Episode #' )
plt.show()

# test the agent
_env = gym.make( 'CartPole-v0' )
## _env = gym.make( 'MountainCar-v0' )

for _ in range( 5 ) :
    _state = _env.reset()
    for t in range( 10000 ) :
        _action, _ = _policy.act( _state )
        _env.render()
        _state, _reward, _done, _ = _env.step( _action )
        if _done:
            print( 'done! t: {:d}'.format( t ) )
            break 

_env.close()

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.optim as opt
import torch.distributions as dst

from IPython.core.debugger import set_trace

TRAIN = False# True
GAMMA = 1.0
LOG_WINDOW = 100

def train( env, model, num_episodes = 2000 ) :
    env.seed( 0 )
    model.train()

    scores_deque = deque( maxlen = LOG_WINDOW )
    mean_score = 0.
    best_score = -np.inf

    optimizer = opt.Adam( model.parameters(), lr = 1e-3 )
    loss_fcn = lambda logprobs, advantages : \
                    torch.stack( [ -log_prob * adv for log_prob, adv in zip( logprobs, advantages )  ] ).sum()

    progressbar = tqdm( range( 1, num_episodes + 1 ), desc = 'Training> ' )

    for iepisode in progressbar :

        _s = env.reset()
        rewards = []
        logprobs = []
        returns = []

        for _ in range( 1000 ) :
            # grab probabilities from model
            _probs = model( torch.from_numpy( _s ).float() )
            # sample an action from the distribution
            _a = dst.Categorical( _probs ).sample().item()
            # take the step in the environment and save transition
            _s, _r, _done, _ = env.step( _a )
            rewards.append( _r )
            logprobs.append( torch.log( _probs[_a] ) )

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

        # define loss and optimize it
        loss = loss_fcn( logprobs, advantages )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iepisode >= LOG_WINDOW :
            mean_score = np.mean( scores_deque )
            message = 'Training> best: %.2f - mean: %.2f'
            progressbar.set_description( message % ( best_score, mean_score ) )
            progressbar.refresh()
        else :
            message = 'Training> best: %.2f'
            progressbar.set_description( message % ( best_score ) )
            progressbar.refresh()

    torch.save( model.state_dict(), './saved/pytorch/vpg_cartpole.pth' )


def test( env, model, num_episodes = 10 ) :
    model.load_state_dict( torch.load( './saved/pytorch/vpg_cartpole.pth' ) )
    model.eval()

    for _ in tqdm( range( num_episodes ), desc = 'Testing> ' ) :
        _done = False
        _s = env.reset()

        while not _done :
            _a = torch.argmax( model( torch.from_numpy( _s ).float() ) ).item()
            _s, _r, _done, _ = env.step( _a )
            env.render()


if __name__ == '__main__' :
    env = gym.make( 'CartPole-v0' )
    
    model = torch.nn.Sequential( nn.Linear( env.observation_space.shape[0], 20 ),
                                 nn.ReLU(),
                                 nn.Linear( 20, env.action_space.n ),
                                 nn.Softmax() )

    if TRAIN :
        train( env, model )
    else :
        test( env, model )

import sys
sys.path.insert( 0, '../' )

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import gridworld
from envs import gridworld_utils
from td_agent import TDPredictionAgent

GAMMA = 1.0
EPSILON = 1.0
ALPHA = 0.1
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000

_env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                               noise = 0.0,
                               rewardAtGoal = -1.0, 
                               rewardAtHole = 0.0,
                               rewardPerStep = -1.0 )

_agent = TDPredictionAgent( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA, True )

for iepisode in tqdm( range( NUM_EPISODES ) ) :

    _done = False
    _state = _env.reset()
    _steps = 0

    while True :

        _action = _env.action_space.sample()
        ## _action = _agent.act( _state, inference = False )

        _snext, _reward, _done, _ = _env.step( _action )
        _transition = ( _state, _action, _reward, _snext, _done )

        _agent.update( _transition )

        if _done :
            break

        if _steps >= MAX_STEPS_PER_EPISODE :
            break

        _steps += 1
        _state = _snext

plt.ion()

print( 'alpha: ', _agent.alpha() )

gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )

_ = input( 'Press ENTER to continue ...' )
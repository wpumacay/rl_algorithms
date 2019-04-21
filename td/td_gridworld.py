
import sys
sys.path.insert( 0, '../' )

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import gridworld
from envs import gridworld_utils
from td_agent import TDPredictionAgent
from td_agent import TDSarsaAgent
from td_agent import TDQlearningAgent
from td_agent import TDExpectedSarsaAgent

GAMMA = 1.0
EPSILON = 1.0
ALPHA = 0.1
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000

_env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                               noise = 0.0,
                               rewardAtGoal = -1.0, 
                               rewardAtHole = 0.0,
                               rewardPerStep = -1.0,
                               randomSeed = 1 )

# TD-Prediction with TD-Learning ###############################################

## _agent = TDPredictionAgent( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA, True )
## 
## for iepisode in tqdm( range( NUM_EPISODES ) ) :
## 
##     _done = False
##     _state = _env.reset()
##     _steps = 0
## 
##     while True :
## 
##         _action = _env.action_space.sample()
##         ## _action = _agent.act( _state, inference = False )
## 
##         _snext, _reward, _done, _ = _env.step( _action )
##         _transition = ( _state, _action, _reward, _snext, _done )
## 
##         _agent.update( _transition )
## 
##         if _done :
##             break
## 
##         if _steps >= MAX_STEPS_PER_EPISODE :
##             break
## 
##         _steps += 1
##         _state = _snext
## 
## plt.ion()
## 
## print( 'alpha: ', _agent.alpha() )
## 
## gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
## 
## _ = input( 'Press ENTER to continue ...' )

# TD-Control with SARSA(0) #####################################################

## _agent = TDSarsaAgent( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA, True )
## 
## for iepisode in tqdm( range( NUM_EPISODES ) ) :
## 
##     _done = False
##     _state = _env.reset()
##     _action = _anext = _agent.act( _state, inference = False )
##     _steps = 0
## 
##     while True :
## 
##         _snext, _reward, _done, _ = _env.step( _action )
##         _anext = _agent.act( _snext, inference = False )
## 
##         _transition = ( _state, _action, _reward, _snext, _anext, _done )
## 
##         _agent.update( _transition )
## 
##         if _done :
##             break
## 
##         if _steps >= MAX_STEPS_PER_EPISODE :
##             break
## 
##         _steps += 1
##         _state = _snext
##         _action = _anext
## 
## plt.ion()
## 
## print( 'alpha: ', _agent.alpha() )
## print( 'epsilon: ', _agent.epsilon() )
## 
## gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
## gridworld_utils.plotQTableInGrid( _agent.Q(), _env.rows, _env.cols )
## 
## _ = input( 'Press ENTER to continue ...' )

# TD-Control with Q-learning (SARSA-MAX) #######################################

## _agent = TDQlearningAgent( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA, True )
## 
## for iepisode in tqdm( range( NUM_EPISODES ) ) :
## 
##     _done = False
##     _state = _env.reset()
##     _steps = 0
## 
##     while True :
## 
##         _action = _agent.act( _state, inference = False )
##         _snext, _reward, _done, _ = _env.step( _action )
## 
##         _transition = ( _state, _action, _reward, _snext, _done )
## 
##         _agent.update( _transition )
## 
##         if _done :
##             break
## 
##         if _steps >= MAX_STEPS_PER_EPISODE :
##             break
## 
##         _steps += 1
##         _state = _snext
## 
## plt.ion()
## 
## print( 'alpha: ', _agent.alpha() )
## print( 'epsilon: ', _agent.epsilon() )
## 
## gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
## gridworld_utils.plotQTableInGrid( _agent.Q(), _env.rows, _env.cols )
## 
## _ = input( 'Press ENTER to continue ...' )

# TD-Control with EXPECTED-SARSA ###############################################

_agent = TDExpectedSarsaAgent( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA, True )

for iepisode in tqdm( range( NUM_EPISODES ) ) :

    _done = False
    _state = _env.reset()
    _steps = 0

    while True :

        _action = _agent.act( _state, inference = False )
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
print( 'epsilon: ', _agent.epsilon() )

gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
gridworld_utils.plotQTableInGrid( _agent.Q(), _env.rows, _env.cols )

_ = input( 'Press ENTER to continue ...' )
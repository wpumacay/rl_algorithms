
import sys
sys.path.insert( 0, '../' )

import numpy as np
import matplotlib.pyplot as plt

from envs import gridworld
from envs import gridworld_utils
from dynprog_agent import *

## _grid = [ [ '.', '.', '.', 'G' ],
##           [ '.', 'B', '.', 'H' ],
##           [ '.', 'B', '.', '.' ],
##           [ '.', '.', '.', '.' ] ]

_env = gridworld.GridWorldEnv( gridworld.DRLBOOTCAMP_CLIFF_LAYOUT,
                               noise = 0.5,
                               rewardAtGoal = 10.0, 
                               rewardAtHole = -10.0,
                               rewardPerStep = 0.0 )
_state = _env.reset()

## # print some of the internals of the gridworld
## print( 'nS: ', _env.nS )
## print( 'nA: ', _env.nA )
## print( 'P: ', _env.P )
## 
## print( 'P[0]: ', _env.P[0] )
## print( 'P[0][0]: ', _env.P[0][0] )
## print( 'P[15]: ', _env.P[15] )
## print( 'P[15][0]: ', _env.P[15][0] )

# Policy Evaluation ############################################################

## Define a policy-function that we want to evaluate

# random policy (returns equal probability for each action)
def randomPolicy( env, state ) :
    return np.ones( env.nA ) / env.nA

# the actual optimal policy (returns 1.0 probability for optimal action)
# this policy corresponds to the gridworld.BOOK_LAYOUT grid layout
def optimalPolicy( env, state ) :
    _nrows = env.rows
    _ncols = env.cols
    _row, _col = env._state2pos( state )

    # define the goal locations
    _goals = [ env._state2pos( sgoal ) for sgoal in [0, 15] ]

    # compute manhattan distance to goals
    _dists = [ np.abs( _goals[i][1] - _row ) + 
               np.abs( _goals[i][0] - _col ) for i in range( len( _goals ) ) ]

    # go to closest goal
    _closestGoal = _goals[ np.argmin( _dists ) ]

    _action = 0

    # try to get closer by rows first, then by cols
    if _closestGoal[0] > _row :
        _action = gridworld.ACTION_DOWN
    elif _closestGoal[0] < _row :
        _action = gridworld.ACTION_UP
    elif _closestGoal[1] > _col :
        _action = gridworld.ACTION_RIGHT
    elif _closestGoal[1] < _col :
        _action = gridworld.ACTION_LEFT
    else :
        _action = env.action_space.sample()

    _probs = np.zeros( env.nA )
    _probs[_action] = 1.0

    return _probs

def testPolicy() :
    # set interactive mode by passing renderInteractive = True to its constructor
    _state = _env.reset()
    _steps = 0
    
    while True :
    
        _aprobs = randomPolicy( _env, _state )
        ## _aprobs = optimalPolicy( _env, _state )
        _action = np.random.choice( _env.nA, p = _aprobs )
    
        _snext, _reward, _done, _ = _env.step( _action )
        _env.render()
    
        print( '**************************' )
        print( 's(t): ', _state )
        print( 'a(t): ', _action )
        print( 's(t+1): ', _snext )
        print( 'r(t+1): ', _reward )
        print( '**************************' )
    
        if _done :
            break
    
        if _steps >= 100 :
            break
    
        _steps += 1
        _state = _snext
    
        _ = input( 'Press any key to continue ...' )

## _gamma = 0.9
## _agent = PolicyEvalAgent( _env, _env.nS, _env.nA, _gamma, randomPolicy )
## 
## _agent.run()
## 
## plt.ion()
## gridworld_utils.plotVTableInGrid( _agent.v, _env.rows, _env.cols )
## gridworld_utils.plotQTableInGrid( _agent.q, _env.rows, _env.cols )
## 
## _ = input( 'Press ENTER to continue ...' )

# ##############################################################################

# Policy Iteration #############################################################

## _state = _env.reset()
## 
## _gamma = 0.9
## _agent = PolicyIterationAgent( _env, _env.nS, _env.nA, _gamma )
## 
## _agent.run()
## 
## plt.close( 'all' )
## plt.ion()
## gridworld_utils.plotVTableInGrid( _agent.v, _env.rows, _env.cols )
## gridworld_utils.plotQTableInGrid( _agent.q, _env.rows, _env.cols )
## 
## _ = input( 'Press ENTER to continue ...' )

# ##############################################################################

# Value Iteration ##############################################################

_state = _env.reset()

_gamma = 0.99
_agent = ValueIterationAgent( _env, _env.nS, _env.nA, _gamma )

_agent.run()

plt.close( 'all' )
plt.ion()
gridworld_utils.plotVTableInGrid( _agent.v, _env.rows, _env.cols )
gridworld_utils.plotQTableInGrid( _agent.q, _env.rows, _env.cols )

_ = input( 'Press ENTER to continue ...' )

# ##############################################################################
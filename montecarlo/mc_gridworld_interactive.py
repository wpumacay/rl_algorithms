
import sys
sys.path.insert( 0, '../' )

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import gridworld
from envs import gridworld_utils

from mc_agent import MCAgentDiscreteFirstVisit

GAMMA = 1.0
EPSILON = 1.0
ALPHA = None
NUM_EPISODES = 10000000
MAX_STEPS_PER_EPISODE = 1000

_env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                               noise = 0.0,
                               rewardAtGoal = -1.0, 
                               rewardAtHole = 0.0,
                               rewardPerStep = -1.0,
                               renderInteractive = True )

_agent = MCAgentDiscreteFirstVisit( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA )

_vviz = gridworld_utils.VTableVisualizer( _env.rows, _env.cols )

for _ in range( NUM_EPISODES ) :

    _done = False
    _episode = []
    _state = _env.reset()
    _agent.beginEpisode()
    _steps = 0

    while True :

        ## _action = _agent.act( _state, inference = False )
        _action = _env.getUserAction()

        if _action == -1 :
            _env.render()
            _vviz.update( _agent.V() )
            continue

        _snext, _reward, _done, _ = _env.step( _action )
        _episode.append( ( _state, _action, _reward ) )

        _env.render()
        _vviz.update( _agent.V() )

        print( '**************************' )
        print( 's(t): ', _state )
        print( 'a(t): ', _action )
        print( 's(t+1): ', _snext )
        print( 'r(t+1): ', _reward )
        print( '**************************' )

        if _done :
            break

        if _steps >= MAX_STEPS_PER_EPISODE :
            break

        _steps += 1
        _state = _snext

    _agent.endEpisode( { 'episode' : _episode } )

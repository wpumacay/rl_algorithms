
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
NUM_EPISODES = 100000
MAX_STEPS_PER_EPISODE = 100000

_env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                               noise = 0.0,
                               rewardAtGoal = 0.0, 
                               rewardAtHole = 0.0,
                               rewardPerStep = -1.0 )

_agent = MCAgentDiscreteFirstVisit( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA )

for _ in tqdm( range( NUM_EPISODES ) ) :

    _done = False
    _episode = []
    _state = _env.reset()
    _agent.beginEpisode()
    _steps = 0

    while True :

        ## _action = _env.action_space.sample()
        _action = np.random.randint( _env.nA )
        _snext, _reward, _done, _ = _env.step( _action )
        #_env.render()
        #_reward = ( GAMMA ** _steps ) * _reward
        _episode.append( ( _state, _action, _reward ) )

        if _done :
            break

        if _steps >= MAX_STEPS_PER_EPISODE :
            break

        _steps += 1
        _state = _snext
        ## print( 's: ', _state, ' a: ', _action, ' r: ', _reward )

    _agent.endEpisode( { 'episode' : _episode } )


print( 'V' )
print( _agent.V() )

plt.ion()

gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
gridworld_utils.plotVisitsInHistogram( _agent.stateVisits(), _env.nS )
gridworld_utils.plotVisitsInGrid( _agent.stateVisits(), _env.rows, _env.cols )

_ = input( 'Press ENTER to continue ...' )
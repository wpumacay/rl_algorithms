
import sys
sys.path.insert( 0, '../' )

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import gridworld
from envs import gridworld_utils
from mc_agent import MCAgentDiscreteFirstVisit
from mc_agent import MCAgentDiscreteEveryVisit

GAMMA = 1.0
EPSILON = 1.0
ALPHA = 0.1
NUM_EPISODES = 1000000
MAX_STEPS_PER_EPISODE = 1000

_env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                               noise = 0.0,
                               rewardAtGoal = -1.0, 
                               rewardAtHole = 0.0,
                               rewardPerStep = -1.0 )

_agent = MCAgentDiscreteFirstVisit( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA )
# _agent = MCAgentDiscreteEveryVisit( _env.nS, _env.nA, GAMMA, EPSILON, ALPHA )

#_hyperParamVizEpsilon = gridworld_utils.HyperparameterScheduleVisualizer( 0, 1 )
#_hyperParamVizAlpha = gridworld_utils.HyperparameterScheduleVisualizer( 0, 1 )

for iepisode in tqdm( range( NUM_EPISODES ) ) :

    _done = False
    _episode = []
    _state = _env.reset()
    _agent.beginEpisode()
    _steps = 0

    while True :

        ## _action = _env.action_space.sample()
        ## _action = np.random.randint( _env.nA )
        _action = _agent.act( _state, inference = False )
        _snext, _reward, _done, _ = _env.step( _action )
        ## _env.render()
        ## _vviz.render( _agent.V() )
        #_reward = ( GAMMA ** _steps ) * _reward
        _episode.append( ( _state, _action, _reward ) )

        if _done :
            break

        if _steps >= MAX_STEPS_PER_EPISODE :
            break

        _steps += 1
        _state = _snext
        
        ## if iepisode % 1000 == 0 :
        ##     #_hyperParamVizEpsilon.update( _agent.epsilon() )
        ##     _hyperParamVizAlpha.update( _agent.alpha() )

    _agent.endEpisode( { 'episode' : _episode } )

_agent.save( 'agent_mc_1_mcalpha' )
## _agent.load( 'agent_mc_0' )

plt.ion()

gridworld_utils.plotVTableInGrid( _agent.V(), _env.rows, _env.cols )
# gridworld_utils.plotVisitsInHistogram( _agent.stateVisits(), _env.nS )
# gridworld_utils.plotVisitsInGrid( _agent.stateVisits(), _env.rows, _env.cols )

gridworld_utils.plotQTableInGrid( _agent.Q(), _env.rows, _env.cols )

print( 'epsilon: ', _agent.epsilon() )

_ = input( 'Press ENTER to continue ...' )

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

# import simple gridworld for testing purposes
from rl.envs import gridworld
from rl.envs import gridworld_utils

# @TODO: Could change this part to have a structure more similar ...
# to the creation mechanisms used in controlsuite (dm_control), ...
# using factory methods that instantiate the functionality required

# import specifics for each environment
## import rl.dqn.dqn_lunarLander as dqn
import rl.dqn.dqn_gridworld as dqn

# import model builder functionality (pytorch as backend)
## import rl.dqn.dqn_model_pytorch as model
import rl.dqn.dqn_model_table as model

from IPython.core.debugger import set_trace

def experiment() :
    MAX_EPISODES = dqn.AGENT_CONFIG.learningMaxSteps
    MAX_STEPS_EPISODE = 100
    LOG_WINDOW_SIZE = 100

    ## _agent = dqn.DqnAgentBuilder( dqn.AGENT_CONFIG,
    ##                               dqn.MODEL_CONFIG,
    ##                               model.DqnModelBuilder )

    _agent = dqn.DqnAgentBuilderTabular( dqn.AGENT_CONFIG,
                                         dqn.MODEL_CONFIG,
                                         model.DqnModelBuilder )

    ## _env = gym.make( 'LunarLander-v2' )
    _env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT,
                                   noise = 0.0,
                                   rewardAtGoal = -1.0,
                                   rewardAtHole = -1.0,
                                   rewardPerStep = -1.0,
                                   renderInteractive = False,
                                   randomSeed = dqn.AGENT_CONFIG.seed )
    # _env.seed( dqn.AGENT_CONFIG.seed )

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _scores = []

    for iepisode in _progressbar :

        _state = _env.reset()
        _score = 0

        for istep in range( MAX_STEPS_EPISODE ) :
            # grab action from dqn agent: runs through model, e-greedy, etc.
            _action = _agent.act( _state, inference = False )
            # apply action in simulator to get the transition
            _snext, _reward, _done, _ = _env.step( _action )
            ## _env.render()
            _transition = ( _state, _action, _snext, _reward, _done )
            # send this transition back to the agent (to learn when he pleases)
            ## set_trace()
            _agent.step( _transition )

            # prepare for next iteration
            _state = _snext
            _score += _reward

            if _done :
                break

        _scores.append( _score )
        _scoresWindow.append( _score )

        if iepisode > LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log results
            if iepisode % LOG_WINDOW_SIZE == 0 :
                ## set_trace()
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr=%.2f, Eps=%.2f' % (_maxAvgScore, _score, _agent.epsilon) )
                _progressbar.refresh()

    # for gridworld, plot the resulting q-table
    plt.ion()

    gridworld_utils.plotQTableInGrid( _agent._qmodel_actor._qtable, _env.rows, _env.cols )
    gridworld_utils.plotQTableInGrid( _agent._qmodel_target._qtable, _env.rows, _env.cols )

    print( 'epsilon: ', _agent.epsilon )

    _ = input( 'Press ENTER to continue ...' )

if __name__ == '__main__' :
    experiment()
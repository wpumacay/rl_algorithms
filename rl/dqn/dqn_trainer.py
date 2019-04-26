
import gym
import numpy as np
from tqdm import tqdm
from collections import deque

# @TODO: Could change this part to have a structure more similar ...
# to the creation mechanisms used in controlsuite (dm_control), ...
# using factory methods that instantiate the functionality required

# import specifics for each environment
import rl.dqn.dqn_lunarLander as dqn

# import model builder functionality (pytorch as backend)
import rl.dqn.dqn_model_pytorch as model

from IPython.core.debugger import set_trace

def experiment() :
    MAX_EPISODES = dqn.AGENT_CONFIG.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    _agent = dqn.DqnAgentBuilder( dqn.AGENT_CONFIG,
                                  dqn.MODEL_CONFIG,
                                  model.DqnModelBuilder )

    _env = gym.make( 'LunarLander-v2' )

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
            _transition = ( _state, _action, _snext, _reward, _done )
            # send this transition back to the agent (to learn when he pleases)
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
            if iepisode % 100 == 0 :
                _progressbar.set_description( 'Training> Max Avg. score: %d' % _maxAvgScore )
                _progressbar.refresh()



if __name__ == '__main__' :
    experiment()
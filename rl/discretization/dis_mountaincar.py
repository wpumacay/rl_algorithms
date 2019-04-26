
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from IPython.core.debugger import set_trace

import dis_utils
import dis_agent

# define some hyperparameters
GAMMA = 0.99
EPSILON = 1.0
ALPHA = 0.02
USE_EPSILON_DECAY = True
USE_ALPHA_DECAY = False
NUM_EPISODES = 10000

def setupGridAgent( slow, shigh, nactions ) :
    # number of bins to use for the discretization
    NUM_BINS = 20
    _nbins = tuple( NUM_BINS for _ in range( len( slow ) ) )
    # create the grid-discretization agent
    _agent = dis_agent.QLearningGridAgent( slow, 
                                           shigh,
                                           nactions,
                                           _nbins,
                                           GAMMA,
                                           EPSILON,
                                           ALPHA,
                                           USE_ALPHA_DECAY,
                                           USE_EPSILON_DECAY )

    return _agent

def setupTilingAgent( slow, shigh, nactions ) :
    # number of bins to use for the discretization
    NUM_BINS = 10
    _nbins = tuple( NUM_BINS for _ in range( len( slow ) ) )
    # offsets for the tilings
    _offsets = ( shigh - slow ) / ( 2 * NUM_BINS )
    # tilings specs
    _tilingSpecs = [ ( _nbins, -_offsets ),
                     ( _nbins, -0.5 * _offsets ),
                     ( _nbins, -0.25 * _offsets ),
                     ( _nbins, tuple( 0.0 for _ in range( len( slow ) ) ) ),
                     ( _nbins, 0.25 * _offsets ),
                     ( _nbins, 0.5 * _offsets ),
                     ( _nbins, _offsets ) ]
    # create the tiling-discretization agent
    _agent = dis_agent.QLearningTilingAgent( slow,
                                             shigh,
                                             nactions,
                                             _tilingSpecs,
                                             GAMMA,
                                             EPSILON,
                                             ALPHA,
                                             USE_ALPHA_DECAY,
                                             USE_EPSILON_DECAY )

    return _agent

def experimentDiscretizationAgent( env, agentType ) :
    # set the seed (for reproducibility)
    env.seed( 505 )
    np.random.seed( 505 )

    # grab low-high limits from environment
    _slow = env.observation_space.low
    _shigh = env.observation_space.high

    # grab the number of discrete actions available
    _nactions = env.action_space.n

    # create the appropriate agent
    _agent = setupGridAgent( _slow, _shigh, _nactions ) if agentType == 'grid' else \
             setupTilingAgent( _slow, _shigh, _nactions )

    # start training
    _progressbar = tqdm( range( 1, NUM_EPISODES + 1 ), desc = 'Training:', leave = True )
    _maxAvgScore = -np.inf
    _scores = []

    for iepisode in _progressbar :
    
        _done = False
        _state = env.reset()
        _steps = 0
        _return = 0.0
    
        while True :
    
            _action = _agent.act( _state, inference = False )
            _snext, _reward, _done, _ = env.step( _action )
            _return += _reward
    
            _transition = ( _state, _action, _reward, _snext, _done )
    
            _agent.update( _transition )
    
            if _done :
                break
    
            _steps += 1
            _state = _snext

        _agent.onEndEpisode()
        _scores.append( _return )

        if len( _scores ) > 100 :
            _avgScore = np.mean( _scores[-100:] )
            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log results
            if iepisode % 100 == 0 :
                _progressbar.set_description( 'Max Avg. score: %d' % _maxAvgScore )
                _progressbar.refresh()

    print( 'alpha: ', _agent.alpha )
    print( 'epsilon: ', _agent.epsilon )

    _ = input( 'Press any key to continue with testing ...' )

    for _ in range( 10 ) :

        _state = env.reset()

        while True :
    
            _action = _agent.act( _state, inference = True )
            _snext, _reward, _done, _ = env.step( _action )
            env.render()
    
            if _done :
                break

            _state = _snext

if __name__ == '__main__' :
    ## experimentDiscretizationAgent( gym.make( 'MountainCar-v0' ), 'grid' )
    ## experimentDiscretizationAgent( gym.make( 'MountainCar-v0' ), 'tiling' )
    experimentDiscretizationAgent( gym.make( 'Acrobot-v1' ), 'tiling' )
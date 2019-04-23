
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import dis_utils
import dis_agent

def experimentGridAgent( env ) :

    # grab low-high limits from environment
    _slow = env.observation_space.low
    _shigh = env.observation_space.high

    # grab the number of discrete actions available
    _nactions = env.action_space.n

    # number of bins to use for the discretization
    _nbins = tuple( 20 for _ in range( len( _slow ) ) )

    # define some hyperparameters
    GAMMA = 0.99
    EPSILON = 1.0
    ALPHA = 0.02
    USE_EPSILON_DECAY = False
    USE_ALPHA_DECAY = False
    NUM_EPISODES = 20000
    MAX_STEPS_PER_EPISODE = 1000

    # create the grid-discretization agent
    _agent = dis_agent.QLearningGridAgent( _slow, 
                                           _shigh,
                                           _nactions,
                                           _nbins,
                                           GAMMA,
                                           EPSILON,
                                           ALPHA,
                                           USE_ALPHA_DECAY,
                                           USE_EPSILON_DECAY )

    # start training
    for iepisode in tqdm( range( NUM_EPISODES ) ) :
    
        _done = False
        _state = env.reset()
        _steps = 0
    
        while True :
    
            _action = _agent.act( _state, inference = False )
            _snext, _reward, _done, _ = env.step( _action )
    
            _transition = ( _state, _action, _reward, _snext, _done )
    
            _agent.update( _transition )
    
            if _done :
                break
    
            if _steps >= MAX_STEPS_PER_EPISODE :
                break
    
            _steps += 1
            _state = _snext

        _agent.onEndEpisode()

    print( 'alpha: ', _agent.alpha )
    print( 'epsilon: ', _agent.epsilon )

    _ = input( 'Press any key to continue with testing ...' )

    for _ in range( 5 ) :

        _state = env.reset()

        while True :
    
            _action = _agent.act( _state, inference = True )
            _snext, _reward, _done, _ = env.step( _action )
            env.render()
    
            if _done :
                break

            _state = _snext

if __name__ == '__main__' :
    experimentGridAgent( gym.make( 'MountainCar-v0' ) )
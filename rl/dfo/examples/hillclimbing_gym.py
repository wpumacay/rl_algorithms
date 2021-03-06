
import os
import sys
import gym
import time
import numpy as np
from tqdm import tqdm
from collections import deque

from IPython.core.debugger import set_trace

from rl.dfo.config import DFOAgentConfig
from rl.dfo.config import DFOModelConfig
from rl.dfo.hillclimbing import HillClimbingAgent

from rl.utils.trainers import SimpleTrainer

BACKEND = 'keras'
TEST = False
MAX_EPISODES = 1000
MAX_EPISODE_LENGTH = 1000
LOG_WINDOW_SIZE = 100
SEED = 0

def train( env, agent ) :
    # seed the env and random number generator
    env.seed( SEED )
    np.random.seed( SEED )

    _scoresBuffer = []
    _scoresAvgs = []
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _maxAvgScore = -np.inf
    _bestScore = -np.inf
    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )

    for iepisode in _progressbar :
        _s = env.reset()
        _score = 0.

        for i in range( 1, MAX_EPISODE_LENGTH + 1 ) :
            _a = agent.act( _s )

            _snext, _r, _done, _ = env.step( _a )

            # send update signal to the agent
            agent.update( (_s, _a, _r, _snext, _done) )

            # book keeping
            _score += _r
            _s = _snext

            if _done :
                break

        # send endEpisode signal to the agent
        agent.onEndEpisode()

        _bestScore = max( _bestScore, _score )

        _scoresBuffer.append( _score )
        _scoresWindow.append( _score )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )

            _scoresAvgs.append( _avgScore )

            _maxAvgScore = max( _maxAvgScore, _avgScore )

            _progressbar.set_description( 'Training> Max-Avg=%.3f, Curr-Avg=%.3f, Curr=%.3f, NoiseScale=%.3f, Best=%.3f' \
                                          % ( _maxAvgScore, _avgScore, _score, agent.noiseScale, _bestScore ) )
            _progressbar.refresh()

            if _avgScore >= 195.0 :
                print( 'Solved environment in %d episodes' % ( iepisode ) )
                break

        else :
            _progressbar.set_description( 'Training> Curr=%.3f, NoiseScale=%.3f, BestScore=%.3f' \
                                          % ( _score, agent.noiseScale, _bestScore ) )
            _progressbar.refresh()            


    test( env, agent )


def test( env, agent ) :
    _progressbar = tqdm( range( 1, 10 + 1 ), desc = 'Testing>', leave = True )
    for _ in _progressbar :

        _state = env.reset()
        _score = 0.0
        _goodBananas = 0
        _badBananas = 0

        while True :
            _action = agent.act( _state )
            _state, _reward, _done, _ = env.step( _action )
            env.render()

            _score += _reward

            if _done :
                break

        _progressbar.set_description( 'Testing> Score=%.2f' % ( _score ) )
        _progressbar.refresh()


def createModelKeras( modelConfig ) :
    try :
        from rl.dfo.model_keras import DFOModelKeras
    except ImportError :
        print( 'ERROR> it seems you don\'t have keras installed in your system' )
        sys.exit( -1 )

    _modelKeras = DFOModelKeras( 'hillclimbing_keras_model', modelConfig )
    _modelKeras.initialize( {} )

    return _modelKeras


def experiment() :
    _env = gym.make( 'CartPole-v0' )

    # model configuration for our experiment
    _modelConfig = DFOModelConfig()
    _modelConfig.inputShape = _env.observation_space.shape if \
                                        isinstance( _env.observation_space, gym.spaces.Box ) else \
                              (_env.observation_space.n,)
    _modelConfig.outputShape = _env.action_space.shape if \
                                        isinstance( _env.action_space, gym.spaces.Box ) else \
                               (_env.action_space.n,)
    _modelConfig.useDiscreteOutputs = isinstance( _env.action_space, gym.spaces.Discrete )
    _modelConfig.layersDefs = [ { 'name' : 'fc1' , 
                                  'type' : 'fc', 
                                  'activation' : 'softmax', 
                                  'useBias' : False, 
                                  'initializer' : 'uniform', 
                                  'initializerArgs' : { 'min' : 0., 'max' : 1e-4, 'seed' : SEED } } ]

    # create the model with the appropriate backend
    if BACKEND == 'keras' :
        _model = createModelKeras( _modelConfig )
    else :
        print( 'ERROR> backend not supported yet, please use either (keras)' )
        sys.exit( -1 )

    # configuration for our experiment
    _agentConfig = DFOAgentConfig()
    _agentConfig.stateSpaceType = 'continuous' if \
                                        isinstance( _env.observation_space, gym.spaces.Box ) else \
                                  'discrete'
    _agentConfig.nStates = _env.observation_space.n if \
                                isinstance( _env.observation_space, gym.spaces.Discrete ) else 0
    _agentConfig.sSize = _env.observation_space.shape if \
                                isinstance( _env.observation_space, gym.spaces.Box ) else (0,)
    _agentConfig.sMin = _env.observation_space.low if \
                                isinstance( _env.observation_space, gym.spaces.Box ) else (-1.,)
    _agentConfig.sMax = _env.observation_space.high if \
                                isinstance( _env.observation_space, gym.spaces.Box ) else (1.,)

    _agentConfig.actionSpaceType = 'continuous' if \
                                        isinstance( _env.action_space, gym.spaces.Box ) else \
                                   'discrete'
    _agentConfig.nActions = _env.action_space.n if \
                                isinstance( _env.action_space, gym.spaces.Discrete ) else 0
    _agentConfig.aSize = _env.action_space.shape if \
                                isinstance( _env.action_space, gym.spaces.Box ) else (0,)
    _agentConfig.aMin = _env.action_space.low if \
                                isinstance( _env.action_space, gym.spaces.Box ) else (-1.,)
    _agentConfig.aMax = _env.action_space.high if \
                                isinstance( _env.action_space, gym.spaces.Box ) else (1.,)
    _agentConfig.gamma = 1.0
    _agentConfig.noiseScale = 1e-2
    _agentConfig.noiseScaleMin = 1e-3
    _agentConfig.noiseScaleMax = 2.0
    _agentConfig.noiseDecayFactor = 0.5
    _agentConfig.noiseGrowthFactor = 2.0
    _agentConfig.useDeterministicPolicy = True

    # create the agent
    _agent = HillClimbingAgent( 'hillclimbing_keras_agent', _agentConfig, _model )

    if TEST :
        test( _env, _agent )
    else :
        train( _env, _agent )


if __name__ == '__main__' :
    experiment()
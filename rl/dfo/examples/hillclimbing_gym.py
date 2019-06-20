
import os
import sys
import gym
import numpy as np
from tqdm import tqdm
from collections import deque

from IPython.core.debugger import set_trace

from rl.dfo.config import DFOAgentConfig
from rl.dfo.config import DFOModelConfig
from rl.dfo.hillclimbing import HillClimbingAgent

BACKEND = 'keras'
TEST = False
MAX_EPISODES = 1000
MAX_EPISODE_LENGTH = 1000
GAMMA = 1.0
LOG_WINDOW_SIZE = 100


def train( env, agent, agentBest ) :
    # seed the env and random number generator
    env.seed( 0 )
    np.random.seed( 0 )

    _scoresBuffer = []
    _scoresAvgs = []
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _stepsWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _maxAvgScore = -np.inf
    _bestScore = -np.inf
    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )

    for iepisode in _progressbar :
        _s = env.reset()
        _score = 0.
        _nsteps = 0

        for i in range( 1, MAX_EPISODE_LENGTH + 1 ) :
            _a = agent.act( _s )

            _snext, _r, _done, _ = env.step( _a )

            _score += _r * ( GAMMA ** i )
            _s = _snext
            _nsteps += 1

            if _done :
                break

        _foundBetter = False
        if _score > _bestScore :
            _foundBetter = True
            _bestScore = _score
            # the best weights are the weights of the current agent
            agentBest.clone( agent )
        else :
            # the best weights are still the previously saved best weights
            agent.clone( agentBest )

        # send endEpisode signal to the agent
        agent.onEndEpisode( { 'foundBetter' : _foundBetter } )

        # apply perturbation for next iteration
        agent.perturb( 'uniform', { 'perturbationScale' : agent.noiseScale } )

        _scoresBuffer.append( _score )
        _scoresWindow.append( _score )
        _stepsWindow.append( _nsteps )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            _avgSteps = np.mean( _stepsWindow )

            _scoresAvgs.append( _avgScore )

            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, NoiseScale=%.2f' \
                                          % ( _maxAvgScore, _avgScore, _score, agent.noiseScale ) )
            _progressbar.refresh()

            if _avgScore >= 195.0 :
                print( 'Solved environment in %d episodes' % ( iepisode ) )
                break


def test( env, agent ) :
    pass


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
    _modelConfig.layersDefs = [ { 'name' : 'fc1' , 'type' : 'fc', 'activation' : 'softmax' } ]

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
    _agentConfig.noiseScale = 1e-2
    _agentConfig.noiseScaleMin = 1e-3
    _agentConfig.noiseScaleMax = 2.0
    _agentConfig.noiseDecayFactor = 0.5
    _agentConfig.noiseGrowthFactor = 2.0
    _agentConfig.useDeterministicPolicy = True

    # create the agent
    _agent = HillClimbingAgent( 'hillclimbing_keras_agent', _agentConfig, _model )
    # create the best agent so far
    _agentBest = HillClimbingAgent( 'hillclimbing_keras_agent_best', _agentConfig, _model )
    # and initialize it to the current agent
    _agentBest.clone( _agent )

    if TEST :
        test( _env, _agent )
    else :
        train( _env, _agent, _agentBest )


if __name__ == '__main__' :
    experiment()
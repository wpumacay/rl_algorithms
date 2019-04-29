
import sys
import gym
import argparse
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
## import rl.dqn.dqn_gridworld as dqn
from rl.dqn import dqn_gridworld
from rl.dqn import dqn_gym_control

# import model builder functionality (pytorch as backend)
# from rl.dqn import dqn_model_pytorch
from rl.dqn import dqn_model_tensorflow
from rl.dqn import dqn_model_table

from IPython.core.debugger import set_trace

TEST = True

def createEnvironment( packageName, domainName ) :
    _env = None

    if packageName == 'gym' :
        _env = gym.make( domainName )
        # sanity check - valid gym envionrments supported
        assert ( type( _env.observation_space ) == gym.spaces.box.Box ), \
               'ERROR> domain %s with non-boxy observation space is not supported' % \
               ( domainName )

        assert ( type( _env.action_space ) == gym.spaces.discrete.Discrete ), \
               'ERROR> domain %s with non-discrete action space is not supported' % \
               ( domainName )

    elif packageName == 'custom' :
        if domainName == 'gridworld' :
            _env = gridworld.GridWorldEnv( gridworld.BOOK_LAYOUT, # DEFAULT_LAYOUT
                                           noise = 0.0,
                                           rewardAtGoal = -1.0, # 10.0
                                           rewardAtHole = -1.0, # -10.0
                                           rewardPerStep = -1.0,
                                           renderInteractive = TEST,
                                           randomSeed = dqn_gridworld.AGENT_CONFIG.seed )
        else :
            print( 'ERROR> Custom domain %s is not supported' % ( domainName ) )

    else :
        print( 'ERROR> package %s is not supported' % ( packageName ) )

    if _env is None :
        print( 'ERROR> something went wrong while creating the environment' )
        sys.exit( 1 )

    return _env

def createAgent( packageName, domainName, env, agentType, library, test = False ) :
    _agent, _savefile = None, None

    if agentType == 'dqn' :
        if packageName == 'gym' :
            # load some parameters from environment
            dqn_gym_control.AGENT_CONFIG.stateDim = env.observation_space.shape[0]
            dqn_gym_control.AGENT_CONFIG.nActions = env.action_space.n
            dqn_gym_control.MODEL_CONFIG.inputShape = ( env.observation_space.shape[0], )
            dqn_gym_control.MODEL_CONFIG.outputShape = ( env.action_space.n, )
            # create the agent using the appropriate factory method
            if library == 'pytorch' :
                _agent = dqn_gym_control.DqnAgentBuilder( dqn_gym_control.AGENT_CONFIG,
                                                          dqn_gym_control.MODEL_CONFIG,
                                                          dqn_model_pytorch.DqnModelBuilder )

                _savefile = 'model_pytorch_dqn_' + domainName + '.pth'
            else :
                _agent = dqn_gym_control.DqnAgentBuilder( dqn_gym_control.AGENT_CONFIG,
                                                          dqn_gym_control.MODEL_CONFIG,
                                                          dqn_model_tensorflow.DqnModelBuilder )

                _savefile = 'model_tensorflow_dqn_' + domainName + '.h5'

        elif packageName == 'custom' and domainName == 'gridworld' :
            # load some parameters from environment
            dqn_gridworld.AGENT_CONFIG.stateDim = env.nS
            dqn_gridworld.AGENT_CONFIG.nActions = env.nA
            dqn_gridworld.MODEL_CONFIG.inputShape = ( env.nS, )
            dqn_gridworld.MODEL_CONFIG.outputShape = ( env.nA, )
            # create the agent using the appropriate factory method
            if library == 'pytorch' :
                _agent = dqn_gridworld.DqnAgentBuilderFapprox( dqn_gridworld.AGENT_CONFIG,
                                                               dqn_gridworld.MODEL_CONFIG,
                                                               dqn_model_pytorch.DqnModelBuilder )

                _savefile = 'model_pytorch_dqn_' + domainName + '.pth'
            else :
                _agent = dqn_gridworld.DqnAgentBuilderFapprox( dqn_gridworld.AGENT_CONFIG,
                                                               dqn_gridworld.MODEL_CONFIG,
                                                               dqn_model_tensorflow.DqnModelBuilder )

                _savefile = 'model_tensorflow_dqn_' + domainName + '.h5'

    elif agentType == 'tabular' : # only for testing with custom gridworld environment
        # load some parameters from environment
        dqn_gridworld.AGENT_CONFIG.stateDim = env.nS
        dqn_gridworld.AGENT_CONFIG.nActions = env.nA
        dqn_gridworld.MODEL_CONFIG.inputShape = ( env.nS, )
        dqn_gridworld.MODEL_CONFIG.outputShape = ( env.nA, )
        # create the agent using the appropriate factory method
        _agent = dqn_gridworld.DqnAgentBuilderTabular( dqn_gridworld.AGENT_CONFIG,
                                                       dqn_gridworld.MODEL_CONFIG,
                                                       dqn_model_table.DqnModelBuilder )

    if _agent is None :
        print( 'ERROR> something went wrong while creating the environment' )
        sys.exit( 1 )

    if test and ( _savefile is not None ) :
        _agent.load( _savefile )

    return _agent, _savefile

def train( env, agent, savefile ) :
    MAX_EPISODES = agent.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    env.seed( agent.seed )

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _scores = []

    for iepisode in _progressbar :

        _state = env.reset()
        _score = 0

        for istep in range( MAX_STEPS_EPISODE ) :
            # grab action from dqn agent: runs through model, e-greedy, etc.
            _action = agent.act( _state, inference = False )
            # apply action in simulator to get the transition
            _snext, _reward, _done, _ = env.step( _action )
            ## env.render()
            _transition = ( _state, _action, _snext, _reward, _done )
            # send this transition back to the agent (to learn when he pleases)
            ## set_trace()
            agent.step( _transition )

            # prepare for next iteration
            _state = _snext
            _score += _reward

            if _done :
                break

        _scores.append( _score )
        _scoresWindow.append( _score )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log results
            if iepisode % LOG_WINDOW_SIZE == 0 :
                ## set_trace()
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr=%.2f, Eps=%.2f' % (_maxAvgScore, _score, agent.epsilon) )
                _progressbar.refresh()

    if savefile is not None :
        agent.save( savefile )

def test( env, agent ) :
    for _ in range( 10 ) :

        _state = env.reset()

        while True :
            _action = agent.act( _state, inference = True )
            _state, _, _done, _ = env.step( _action )
            env.render()

            if _done :
                break

def experiment( packageName, domainName, agentType, library ) :

    _env = createEnvironment( packageName, domainName )

    if not TEST :
        _agent, _savefile = createAgent( packageName, domainName, _env, agentType, library )
        train( _env, _agent, _savefile )

        if domainName == 'gridworld' and agentType == 'tabular' :
            # for gridworld, plot the resulting q-table
            plt.ion()
        
            gridworld_utils.plotQTableInGrid( _agent._qmodel_actor._qtable, _env.rows, _env.cols )
            gridworld_utils.plotQTableInGrid( _agent._qmodel_target._qtable, _env.rows, _env.cols )

        print( 'epsilon: ', _agent.epsilon )

        test( _env, _agent )

    else :
        _agent, _savefile = createAgent( packageName, domainName, _env, agentType, library, test = True )
        test( _env, _agent )

    _ = input( 'Press ENTER to continue ...' )

if __name__ == '__main__' :
    _parser = argparse.ArgumentParser()
    _parser.add_argument( 'package_name', help='package to use for simulation [gym|ale|custom]' )
    _parser.add_argument( 'domain_name', help='domain to load from the package [LunarLander-v2|SpaceInvaders|gridworld]' )
    _parser.add_argument( 'agent_type', help='type of agent to be used [dqn|ppo|ddpg|tabular(for gridworl only)]' )
    _parser.add_argument( 'mode', help='mode to run the experiment (train|test)', type=str, choices=['train', 'test'] )
    _parser.add_argument( '--library', help='deep learning library to use (pytorch|tensorflow)', type=str, choices=['tensorflow','pytorch'], default='pytorch' )

    _args = _parser.parse_args()

    print( '#############################################################' )
    print( '#                                                           #' )
    print( '#            Environment and agent setup                    #' )
    print( '#                                                           #' )
    print( '#############################################################' )
    print( 'Package     : ', _args.package_name )
    print( 'Domain      : ', _args.domain_name )
    print( 'Agent       : ', _args.agent_type )
    print( 'Mode        : ', _args.mode )
    print( 'Library     : ', _args.library )
    print( '#############################################################' )

    TEST = ( _args.mode == 'test' )
    experiment( _args.package_name, _args.domain_name, _args.agent_type, _args.library )
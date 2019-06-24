
import gin
import sys
import numpy as np
import gym

from IPython.core.debugger import set_trace

from rl.utils.config import TrainerConfig
from rl.utils.trainers import SimpleTrainer

from rl.dfo.config import DFOAgentConfig
from rl.dfo.config import DFOModelConfig

from rl.dfo.hillclimbing import HillClimbingAgent


if __name__ == '__main__' :
    gin.parse_config_file( 'hillclimbing_config.gin' )

    # create trainer configuration, with params filled by gin-config
    _trainerConfig = TrainerConfig()

    # create the environment ###################################################
    _env = None
    _envType = _trainerConfig.envType
    _envName = _trainerConfig.envName

    if _envType == 'gym' :
        try :
            _env = gym.make( _envName )
        except gym.error.UnregisteredEnv :
            print( 'ERROR> env. %s not found in gym library' % _envName )
            sys.exit( -1 )
    else :
        print( 'ERROR> env. library %s not supported' % _envType )
        sys.exit( -1 )

    ############################################################################

    # create the agent #########################################################
    _agent = None
    _model = None
    # create agent configuration, with params filled by gin-config
    _agentConfig = DFOAgentConfig()
    # create model configuration, with params filled by gin-config
    _modelConfig = DFOModelConfig()

    # configure both model and agent according to the gym environment
    if _envType == 'gym' :
        # grab input shape from gym environment's observation space accordingly
        _modelConfig.inputShape = _env.observation_space.shape if \
                                            isinstance( _env.observation_space, gym.spaces.Box ) else \
                                  (_env.observation_space.n,)

        # grab output shape from gym environment's observation space accordingly
        _modelConfig.outputShape = _env.action_space.shape if \
                                            isinstance( _env.action_space, gym.spaces.Box ) else \
                                   (_env.action_space.n,)

        # grab the type of output of our model accordingly
        _modelConfig.useDiscreteOutputs = isinstance( _env.action_space, gym.spaces.Discrete )

        # grab the type of state-space used in the environment
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
    
        # grab the type of action-space used in the environment
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

    if _trainerConfig.dlbackend == 'keras' :
        from rl.dfo.model_keras import DFOModelKeras
        _model = DFOModelKeras( 'hillclimbing_keras_model', _modelConfig )
        _model.initialize()
    else :
        print( 'ERROR> backend (%s) not supported yet, please use another (keras|)' \
               % _trainerConfig.dlbackend )
        sys.exit( -1 )

    # construct the agent with the already created model
    _agent = HillClimbingAgent( 'agent0', _agentConfig, _model )

    ############################################################################

    _trainer = SimpleTrainer( _trainerConfig, _env, _agent )
    _trainer.init()

    if _trainer.mode == 'train' :
        print( 'Starting training ...' )
        _trainer.train()
        print( 'Finished training.' )

        if _trainer.testOnceTrained :
            print( 'Starting testing ...' )
            _trainer.test()
            print( 'Finished testing.' )

    else :
        print( 'Starting testing ...' )
        _trainer.test()
        print( 'Finished testing.' )
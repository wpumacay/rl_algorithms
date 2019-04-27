
from rl.dqn.core.dqn_agent_interface import IDqnAgent
from rl.dqn.utils.dqn_utils import DqnAgentConfig 
from rl.dqn.utils.dqn_utils import DqnModelConfig

class DqnGymControlAgent( IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder ) :
        super( DqnGymControlAgent, self ).__init__( agentConfig, modelConfig, modelBuilder )

    def _preprocess( self, rawState ) :
        """Default preprocessing by just copying the data

        Args:
            rawState (np.ndarray) : raw state from lunar lander environment

        Returns:
            np.ndarray : copy of the gym-env. observation for the model

        """
        return rawState.copy()

DqnAgentBuilder = lambda agentConfig, modelConfig, modelBuilder : \
                                DqnGymControlAgent( agentConfig, modelConfig, modelBuilder )

AGENT_CONFIG = DqnAgentConfig()
AGENT_CONFIG.stateDim                   = -1 # written by trainer using env. info
AGENT_CONFIG.nActions                   = -1 # written by trainer using env. info
AGENT_CONFIG.epsilonSchedule            = 'geometric'
AGENT_CONFIG.epsilonStart               = 1.0
AGENT_CONFIG.epsilonEnd                 = 0.1
AGENT_CONFIG.epsilonDecay               = 0.995
AGENT_CONFIG.lr                         = 0.0005
AGENT_CONFIG.minibatchSize              = 64
AGENT_CONFIG.learningStartsAt           = 0
AGENT_CONFIG.learningUpdateFreq         = 4
AGENT_CONFIG.learningUpdateTargetFreq   = 4
AGENT_CONFIG.learningMaxSteps           = 2000
AGENT_CONFIG.replayBufferSize           = 10000
AGENT_CONFIG.discount                   = 0.99
AGENT_CONFIG.tau                        = 0.001
AGENT_CONFIG.seed                       = 0

MODEL_CONFIG = DqnModelConfig()
MODEL_CONFIG.inputShape = ( -1, )  # replaced by the trainer using env. info
MODEL_CONFIG.outputShape = ( -1, ) # replaced by the trainer using env. info
MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                        { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                        { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                        { 'name': 'fc4', 'type' : 'fc', 'units' : -1, 'activation' : 'linear' } ]



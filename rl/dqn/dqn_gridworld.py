
import numpy as np

from rl.dqn.core.dqn_agent_interface import IDqnAgent
from rl.dqn.utils.dqn_utils import DqnAgentConfig 
from rl.dqn.utils.dqn_utils import DqnModelConfig

class DqnGridworldAgentTabular( IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder ) :
        super( DqnGridworldAgentTabular, self ).__init__( agentConfig, modelConfig, modelBuilder )

    def _preprocess( self, rawState ) :
        # raw state is state indx, so just return it for tabular case
        return rawState

class DqnGridworldAgentFapprox( IDqnAgent ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder ) :
        super( DqnGridworldAgentFapprox, self ).__init__( agentConfig, modelConfig, modelBuilder )

    def _preprocess( self, rawState ) :
        # rawState is an index, so convert it to a one-hot representation
        _stateOneHot = np.zeros( self._stateDim )
        _stateOneHot[rawState] = 1.0

        return _stateOneHot

DqnAgentBuilderTabular = lambda agentConfig, modelConfig, modelBuilder : \
                            DqnGridworldAgentTabular( agentConfig, modelConfig, modelBuilder )

DqnAgentBuilderFapprox = lambda agentConfig, modelConfig, modelBuilder : \
                            DqnGridworldAgentFapprox( agentConfig, modelConfig, modelBuilder )

AGENT_CONFIG = DqnAgentConfig()
AGENT_CONFIG.stateDim                   = 16 # gridworld book_layout
AGENT_CONFIG.nActions                   = 4
AGENT_CONFIG.epsilonSchedule            = 'geometric'
AGENT_CONFIG.epsilonStart               = 1.0
AGENT_CONFIG.epsilonEnd                 = 0.01
AGENT_CONFIG.epsilonDecay               = 0.99995
AGENT_CONFIG.lr                         = 0.01
AGENT_CONFIG.minibatchSize              = 32
AGENT_CONFIG.learningStartsAt           = 0
AGENT_CONFIG.learningUpdateFreq         = 4
AGENT_CONFIG.learningUpdateTargetFreq   = 16
AGENT_CONFIG.learningMaxSteps           = 100000
AGENT_CONFIG.replayBufferSize           = 200
AGENT_CONFIG.discount                   = 0.99
AGENT_CONFIG.tau                        = 0.1
AGENT_CONFIG.seed                       = 0

MODEL_CONFIG = DqnModelConfig()
MODEL_CONFIG.inputShape = ( 16, ) # rank-1 tensor ( 16-vector )
MODEL_CONFIG.outputShape = ( 4, ) # rank-1 tensor ( 4-vector, 1 qvalue per action )
MODEL_CONFIG.layers = [ { 'name': 'fc1', 'type' : 'fc', 'units' : 128, 'activation' : 'relu' },
                        { 'name': 'fc2', 'type' : 'fc', 'units' : 64, 'activation' : 'relu' },
                        { 'name': 'fc3', 'type' : 'fc', 'units' : 16, 'activation' : 'relu' },
                        { 'name': 'fc4', 'type' : 'fc', 'units' : 64, 'activation' : 'linear' } ]



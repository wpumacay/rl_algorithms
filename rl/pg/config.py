
import gin
import numpy as np

@gin.configurable
class PGAgentConfig( object ) :

    def __init__( self,
                  stateSpaceShape = (2,),
                  stateSpaceRangeMin = (-1., -1.) ,
                  stateSpaceRangeMax = (1., 1.),
                  actionSpaceType = 'discrete',
                  actionSpaceShape = (2,),
                  actionSpaceRangeMin = (-1., -1.),
                  actionSpaceRangeMax = (1., 1.),
                  gamma = 1.0 ):
        super( PGAgentConfig, self ).__init__()

        # description of the state space
        self.stateSpaceShape = stateSpaceShape
        self.stateSpaceRangeMin = stateSpaceRangeMin
        self.stateSpaceRangeMax = stateSpaceRangeMax

        # description of the action space
        self.actionSpaceType = actionSpaceType
        self.actionSpaceShape = actionSpaceShape
        self.actionSpaceShape = actionSpaceShape
        self.actionSpaceRangeMin = actionSpaceRangeMin
        self.actionSpaceRangeMax = actionSpaceRangeMax

        # discount factor
        self.gamma = gamma


@gin.configurable
class PGModelConfig( object ) :

    def __init__( self,
                  inputShape = (2,),
                  outputShape = (2,),
                  layersDefs = [] ) :
        super( PGModelConfig, self ).__init__()

        # input and output sizes of the network
        self.inputShape = inputShape
        self.outputShape = outputShape

        # layers definitions to be used for backend instantiation
        self.layersDefs = layersDefs
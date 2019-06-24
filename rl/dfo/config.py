
import gin
import numpy as np

@gin.configurable
class DFOAgentConfig( object ) :

    def __init__( self,
                  stateSpaceType = 'continuous',
                  nStates = 2,
                  sSize = (2,),
                  sMin = (-1., -1.) ,
                  sMax = (1., 1.),
                  actionSpaceType = 'discrete',
                  nActions = 2,
                  aSize = (2,),
                  aMin = (-1., -1.),
                  aMax = (1., 1.),
                  eps0 = 0.1,
                  sigma0 = 0.1,
                  noiseScale = 1e-2,
                  noiseScaleMin = 1e-3,
                  noiseScaleMax = 2.0,
                  noiseDecayFactor = 0.5,
                  noiseGrowthFactor = 2.0,
                  useDeterministicPolicy = True,
                  gamma = 1.0 ):
        super( DFOAgentConfig, self ).__init__()

        self.stateSpaceType = stateSpaceType
        # if state space is discrete
        self.nStates = nStates
        # if state space is continuous
        self.sSize = sSize
        self.sMin = sMin
        self.sMax = sMax

        self.actionSpaceType = actionSpaceType
        # if action space is discrete
        self.nActions = nActions
        # if action space is continuous
        self.aSize = aSize
        self.aMin = aMin
        self.aMax = aMax

        # parameters for sampling perturbations from a gaussian
        self.eps0 = eps0
        self.sigma0 = sigma0

        # parameters for perturbations from uniform distribution
        self.noiseScale = noiseScale
        self.noiseScaleMin = noiseScaleMin
        self.noiseScaleMax = noiseScaleMax
        self.noiseDecayFactor = noiseDecayFactor
        self.noiseGrowthFactor = noiseGrowthFactor

        # for hill-climbing, if actions are deterministic or stochastic
        self.useDeterministicPolicy = useDeterministicPolicy

        # discount factor
        self.gamma = gamma


@gin.configurable
class DFOModelConfig( object ) :

    def __init__( self,
                  inputShape = (2,),
                  outputShape = (2,),
                  useDiscreteOutputs = True,
                  layersDefs = [ # first layer
                                 { 'name' : 'hidden_fc1' , 
                                   'type' : 'fc', 
                                   'units' : 512, 
                                   'activation' : 'relu', 
                                   'useBias' : True, 
                                   'initializer' : 'uniform', 
                                   'initializerArgs' : { 'min' : 0., 'max' : 1. } },
                                 # second|last layer
                                 { 'name' : 'out_fc2' , 
                                   'type' : 'fc', 
                                   'activation' : 'tanh' } ] ) :
        super( DFOModelConfig, self ).__init__()

        # input and output sizes of the network
        self.inputShape = inputShape
        self.outputShape = outputShape

        # whether or not output probs, or mean of a gaussian
        self.useDiscreteOutputs = useDiscreteOutputs

        # layers definitions to be used for backend instantiation
        self.layersDefs = layersDefs
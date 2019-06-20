
import numpy as np


class DFOAgentConfig( object ) :

    def __init__( self ):
        super( DFOAgentConfig, self ).__init__()

        self.stateSpaceType = 'continuous'
        # if state space is discrete
        self.nStates = 2
        # if state space is continuous
        self.sSize = (2,)
        self.sMin = (-1., -1.)
        self.sMax = (1., 1.)

        self.actionSpaceType = 'discrete'
        # if action space is discrete
        self.nActions = 2
        # if action space is continuous
        self.aSize = (2,)
        self.aMin = (-1., -1.)
        self.aMax = (1., 1.)

        # parameters for sampling perturbations from a gaussian
        self.eps0 = 0.1
        self.sigma0 = 0.1

        # parameters for perturbations from uniform distribution
        self.noiseScale = 1e-2
        self.noiseScaleStart = 1e-2
        self.noiseScaleMin = 1e-3
        self.noiseScaleMax = 2.0
        self.noiseDecayFactor = 0.5
        self.noiseGrowthFactor = 2.0

        # for hill-climbing, if actions are deterministic or stochastic
        self.useDeterministicPolicy = True


class DFOModelConfig( object ) :

    def __init__( self ) :
        super( DFOModelConfig, self ).__init__()

        # input and output sizes of the network
        self.inputShape = (2,)
        self.outputShape = (2,)

        # whether or not output probs, or mean of a gaussian
        self.useDiscreteOutputs = True

        # layers definitions to be used for backend instantiation
        self.layersDefs = [ { 'name' : 'hidden_fc1' , 'type' : 'fc', 'units' : 512, 'activation' : 'relu' },
                            { 'name' : 'out_fc2' , 'type' : 'fc', 'activation' : 'tanh' } ]
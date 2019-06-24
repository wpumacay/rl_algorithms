
import numpy as np
from rl.dfo.agent import DFOAgent

from IPython.core.debugger import set_trace

ERROR_MSG_KEY_ERROR = 'ERROR> key (%s) should have been provided'

class HillClimbingAgent( DFOAgent ) :

    def __init__( self, name, config, model ) :
        super( HillClimbingAgent, self ).__init__( name, config, model )

        # scale factor for the uniform noise applied to the weights
        self._noiseScale = config.noiseScale
        # score of a trajectory
        self._score = 0.
        # best score so far
        self._bestScore = -np.inf
        # discount factor
        self._gamma = config.gamma
        # step counter
        self._istep = 0

        # best model so far (clone the current model)
        self._bestModel = self._model.clone( self._model.name + '_best' )

    def onStartEpisode( self, args = {} ) :
        pass # do nothing at the beginning of the episode

    def update( self, transition ) :
        # unpack the reward from the transition ( s, a, r, s', done )
        _, _, _r, _, _ = transition
        # update the score, discounting it appropriately
        self._score += (self._gamma ** self._istep) * _r
        # and also update the step counter
        self._istep += 1


    def onEndEpisode( self, args = {} ) :
        # check if found a better solution
        if self._score >= self._bestScore :
            # update the best score so far
            self._bestScore = self._score

            # update the best weights from the current weights 
            self._bestModel.copy( self._model )

            # update noise factor (reduce it to narrow our search area)
            self._noiseScale = max( self._config.noiseScaleMin, \
                                    self._noiseScale * self._config.noiseDecayFactor )
        else :
            # roll back to the previous best weights
            self._model.copy( self._bestModel )

            # update noise factor (increase it to search a larger area)
            self._noiseScale = min( self._config.noiseScaleMax, \
                                    self._noiseScale * self._config.noiseGrowthFactor )

        # perturb the current model for a new candidate
        self.perturb( 'uniform', { 'perturbationScale' : self._noiseScale } )

        # clear counters and accumulators for the next iteration
        self._istep = 0
        self._score = 0.


    @property
    def noiseScale( self ) :
        return self._noiseScale
    
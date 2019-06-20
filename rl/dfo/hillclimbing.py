
import numpy as np
from rl.dfo.agent import DFOAgent

ERROR_MSG_KEY_ERROR = 'ERROR> key (%s) should have been provided'

class HillClimbingAgent( DFOAgent ) :

    def __init__( self, name, config, model ) :
        super( HillClimbingAgent, self ).__init__( name, config, model )

        self._noiseScale = config.noiseScale


    def act( self, state ) :
        if self._config.actionSpaceType == 'discrete' :
            # model should return probabilities of each action
            _actProbs = self._model.eval( state[np.newaxis,...] )
            if self._config.useDeterministicPolicy :
                return np.argmax( _actProbs )
            else :
                return np.random.choice( self._config.nActions, p = _actProbs )
        else :
            # the output comes from a gaussian, with mean given by model output
            return self._model.eval( state )


    def onEndEpisode( self, args ) :
        assert 'foundBetter' in args, ERROR_MSG_KEY_ERROR % ('foundBetter',)

        _foundBetter = args['foundBetter']
        if _foundBetter :
            self._noiseScale = max( self._config.noiseScaleMin, \
                                    self._noiseScale * self._config.noiseDecayFactor )
        else :
            self._noiseScale = min( self._config.noiseScaleMax, \
                                    self._noiseScale * self._config.noiseGrowthFactor )


    @property
    def noiseScale( self ) :
        return self._noiseScale
    
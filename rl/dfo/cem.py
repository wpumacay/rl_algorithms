
import numpy as np
from rl.dfo.agent import DFOAgent

from IPython.core.debugger import set_trace

ERROR_MSG_KEY_ERROR = 'ERROR> key (%s) should have been provided'

class CEMAgent( DFOAgent ) :

    def __init__( self, name, config, model ) :
        super( CEMAgent, self ).__init__( name, config, model )
        

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
    
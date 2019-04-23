

import numpy as np
import dis_utils as utils

from IPython.core.debugger import set_trace




class IQFunction( object ) :

    def __init__( self, stateSpaceDim, numActions ):
        super( IQFunction, self ).__init__()

        self._dimS = stateSpaceDim
        self._nA = numActions

    def eval( self, state ) :
        raise NotImplementedError( 'IQFunction::eval> virtual method' )

    def update( self, transition ) :
        raise NotImplementedError( 'IQFunction::update> virtual method' )

    def updateBatch( self, transitions ) :
        raise NotImplementedError( 'IQFunction::updateBatch> virtual method' )

    @property
    def dimS( self ) :
        return self._dimS
    
    @property
    def nA( self ) :
        return self._nA


class QFunctionTiling( IQFunction ) :

    def __init__( self, stateSpaceDim, lowSSpace, highSSpace, numActions ) :
        super( QFunctionTiling, self ).__init__( stateSpaceDim, numActions )

        # the model used
        self._table = utils.TilingTable()



import random
import numpy as np

class MSimpleMPstate( object ) :
    
    def __init__( self, name, terminal = False ) :
        self.name = name
        self.terminal = terminal
         # list of ( state, prob ), probs sum ...
         # to 1.0, if not, normalize
        self.connections = []

class MSimpleMP( object ) :

    def __init__( self, states ) :
        self.m_states = states
        self.m_transitionMatrix = None
        self.m_spaceDim = len( states )
        self.m_indexTable = {}
        self.m_nameTable = {}
        self.m_timestep = 0

        self.m_currentState = None

        self._build()

    def _build( self ) :
        # build the index mapping
        _indx = 0
        for _name in self.m_states :
            self.m_indexTable[ _name ] = _indx
            self.m_nameTable[ _indx ] = _name
            _indx += 1

        # create transition matrix
        self.m_transitionMatrix = np.zeros( ( self.m_spaceDim, 
                                              self.m_spaceDim ), 
                                            dtype = np.float )
        # fill every column from the transition probs
        for i in range( self.m_spaceDim ) :
            _state = self.m_states[ self.m_nameTable[i] ]
            for ( neighbour, prob ) in _state.connections :
                j = self.m_indexTable[ neighbour.name ]
                self.m_transitionMatrix[ j, i ] = prob

    def _pickRandomConnection( self, connections ) :
        _randVal = random.uniform( 0, 1 )
        _cumProb = 0.0

        for ( neighbour, prob ) in connections :
            _cumProb += prob
            if _randVal < _cumProb : 
                return neighbour

        print( 'ERROR> Should not get here' )
        return None

    def run( self, initialState, nsteps = 10 ) :
        self.m_timestep = 0
        self.m_currentState = initialState
        _episode = [ self.m_currentState.name ]

        for _ in range( nsteps ) :
            if self.m_currentState.terminal :
                break
                
            self.m_currentState = self.step( self.m_currentState )
            _episode.append( self.m_currentState.name )

            self.m_timestep += 1
        
        return _episode

    def runChain( self, initialDistribution, nsteps = 10 ) :
        _sdist = np.zeros( ( self.m_spaceDim, 1 ) )
        for i in range( self.m_spaceDim ) :
            _sdist[i] = initialDistribution[i]
        
        for _ in range( nsteps ) :
            _sdist = np.dot( self.m_transitionMatrix, _sdist )

        return _sdist

    def step( self, state ) :
        return self._pickRandomConnection( state.connections )

    def transitionMatrix( self ) : 
        return self.m_transitionMatrix

import random
import numpy as np

class MSimpleMRPstate( object ) :

    def __init__( self, name, terminal = False ) :
        self.name = name
        self.terminal = terminal
         # list of ( state, prob, reward ), probs sum ...
         # to 1.0, if not, normalize
        self.connections = []

class MSimpleMRP( object ) :

    def __init__( self, states, gamma = 0.9 ) :
        self.m_states = states
        self.m_transitionMatrix = None
        self.m_rewardVector = None
        self.m_spaceDim = len( states )
        self.m_indexTable = {}
        self.m_nameTable = {}
        self.m_timestep = 0
        self.m_gamma = gamma

        self.m_currentState = None

        self._build()

    def _build( self ) :
        # build the index mapping
        _indx = 0
        for _name in self.m_states :
            self.m_indexTable[ _name ] = _indx
            self.m_nameTable[ _indx ] = _name
            _indx += 1

        # create transition matrix and reward vector
        self.m_transitionMatrix = np.zeros( ( self.m_spaceDim, 
                                              self.m_spaceDim ), 
                                            dtype = np.float )
        self.m_rewardVector = np.zeros( ( self.m_spaceDim, 1 ), 
                                        dtype = np.float )
        # fill every column from the transition probs
        for i in range( self.m_spaceDim ) :
            _state = self.m_states[ self.m_nameTable[i] ]
            _expReward = 0.0
            for ( neighbour, prob, reward ) in _state.connections :
                j = self.m_indexTable[ neighbour.name ]
                self.m_transitionMatrix[ j, i ] = prob
                _expReward += prob * reward
            
            self.m_rewardVector[i] = _expReward

    def _pickRandomConnection( self, connections ) :
        _randVal = random.uniform( 0, 1 )
        _cumProb = 0.0

        for ( neighbour, prob, reward ) in connections :
            _cumProb += prob
            if _randVal < _cumProb : 
                return neighbour, reward

        print( 'ERROR> Should not get here' )
        return None, 0.0

    def run( self, initialState, nsteps = 10 ) :
        self.m_timestep = 0
        self.m_currentState = initialState

        _return = 0.0
        _episode = [ self.m_currentState.name, 0.0 ]

        for _ in range( nsteps ) :
            if self.m_currentState.terminal :
                break
                
            self.m_currentState, _r = self.step( self.m_currentState )
            _episode.append( ( self.m_currentState.name, 
                               _r * ( self.m_gamma ** self.m_timestep ) ) )

            _return += _r * ( self.m_gamma ** self.m_timestep )

            self.m_timestep += 1

        return _episode, _return

    def runChain( self, initialDistribution, nsteps = 10 ) :
        _sdist = np.zeros( ( self.m_spaceDim, 1 ) )
        for i in range( self.m_spaceDim ) :
            _sdist[i] = initialDistribution[i]
        
        for _ in range( nsteps ) :
            _sdist = np.dot( self.m_transitionMatrix, _sdist )

        return _sdist

    def computeValueFunction( self ) :
        A = np.eye( self.m_spaceDim ) - self.m_gamma * self.m_transitionMatrix
        b = self.m_rewardVector
        return np.linalg.solve( A.T, b )

    def step( self, state ) :
        return self._pickRandomConnection( state.connections )

    def transitionMatrix( self ) : 
        return self.m_transitionMatrix

    def rewardVector( self ) :
        return self.m_rewardVector

import random
import numpy as np

class MSimpleMDPstate( object ) :

    def __init__( self, name, terminal = False, chanceNode = False ) :
        self.name = name
        self.terminal = terminal
        self.chanceNode = chanceNode
        # form [ (neighbour, prob, reward), ... ]
        self.connections = []

class MSimplePolicyInterface( object ) :

    def __init__( self, name, mdp ) :
        self.m_mdp = mdp

    def getAction( self, state ) :
        return None

class MSimpleMDP( object ) :

    def __init__( self, states, actions, gamma = 0.9 ) :
        # form { name : state, ... }
        self.m_states = states
        # form { aname : { sname : neighbour, ... }, ... }
        self.m_actions = actions
        self.m_currentState = None
        self.m_spaceDim = len( states )
        self.m_timestep = 0
        self.m_gamma = gamma

    def states( self ) : 
        return self.m_states
    
    def actions( self ) :
        return self.m_actions

    def _moveChanceNode( self, state ) :
        _randVal = random.uniform( 0, 1 )
        _cumProb = 0.0

        for ( neighbour, prob, reward ) in state.connections :
            _cumProb += prob
            if _randVal < _cumProb : 
                return neighbour, reward

        print( 'ERROR> Should not get here' )
        return None, 0.0

    def run( self, initialState, policy, maxsteps = 10 ) :
        self.m_timestep = 0
        self.m_currentState = initialState

        _return = 0.0
        _episode = [ self.m_currentState.name, 0.0 ]

        for _ in range( maxsteps ) :
            if self.m_currentState.terminal :
                break

            if self.m_currentState.chanceNode :
                self.m_currentState, _r = self._moveChanceNode( self.m_currentState )
            else :                
                _action = policy.getAction( self.m_currentState )
                self.m_currentState, _r = self.step( self.m_currentState,
                                                    _action )

            _episode.append( ( self.m_currentState.name, 
                               _r * ( self.m_gamma ** self.m_timestep ) ) )

            _return += _r * ( self.m_gamma ** self.m_timestep )

            self.m_timestep += 1

        return _episode, _return

    def step( self, state, action ) :
        # check if the action can be executed in this state
        if state.name not in action :
            print( 'Cant execute this action from this state' )
            return state, 0.0

        _neighbour = action[ state.name ]
        # check if this neighbour is in the states dict
        if _neighbour.name not in self.m_states :
            print( 'The resulting state is not a valid state' )
            return state, 0.0

        # check if this neighbour is a valid neighbour for this state
        for (_sneighbour, _, _reward ) in state.connections :
            if _sneighbour.name == _neighbour.name :
                return _sneighbour, _reward
        
        print( 'The action resulting state is not the same as the dynamics' )
        return state, 0.0
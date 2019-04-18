
import numpy as np
import matplotlib.pyplot as plt

from gym import Env, spaces
from gym.utils import seeding

CELL_EMPTY      = 0
CELL_BLOCKED    = 1
CELL_HOLE       = 2
CELL_GOAL       = 3

CELL_CHAR_MAP = { '.' : CELL_EMPTY, 
                  'B' : CELL_BLOCKED, 
                  'H' : CELL_HOLE, 
                  'G' : CELL_GOAL }

CELL_COLORS = [ [ 0.0, 1.0, 0.0 ],
                [ 0.0, 0.0, 0.0 ],
                [ 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 1.0 ],
                [ 0.5, 0.5, 0.5 ] ]

COLOR_CURRENT_POSITION = [ 0.5, 0.5, 0.5 ]

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

BOOK_LAYOUT = [ [ 'G', '.', '.', '.' ],
                [ '.', '.', '.', '.' ],
                [ '.', '.', '.', '.' ],
                [ '.', '.', '.', 'G' ] ]

DEFAULT_LAYOUT = [ [ '.', '.', '.', '.' ],
                   [ '.', 'H', '.', 'H' ],
                   [ '.', '.', '.', 'H' ],
                   [ 'H', '.', '.', 'G' ] ] # frozenlake-v0 layour

class GridWorldEnv( Env ) :

    def __init__( self, 
                  gridLayout = None, 
                  noise = 0.1,
                  rewardAtGoal = 1.0, 
                  rewardAtHole = -1.0,
                  rewardPerStep = 0.0 ) :
        super( GridWorldEnv, self ).__init__()

        # layout for this gridworld, of the form :
        # [ [ '.', '.', '.', 'G' ],
        #   [ '.', 'B', '.', 'H' ],
        #   [ '.', 'B', '.', '.' ],
        #   [ 'S', '.', '.', '.' ] ]
        # . : free cell
        # B : blocked cell
        # H : hole cell
        # G : goal/terminal cell
        self.m_gridLayout = gridLayout if ( gridLayout is not None ) else DEFAULT_LAYOUT

        self.m_grid = None
        self.m_rows = 0
        self.m_cols = 0

        self.m_currentState = 0 # top-left
        # transition model P( s', r | s, a ) in the form ...
        # P = { s : { a : [ ( p, nextState, reward, done ) ] } }
        self.m_transitionModel = {}
        # initial transition distribution
        self.m_isd = []

        # random state and seed
        self.m_seed = 0
        self.m_randomState = None

        # prob of going sideways
        self.m_noise = noise
        # timestep
        self.m_timestep = 0

        self.m_rewardAtGoal = rewardAtGoal
        self.m_rewardAtHole = rewardAtHole
        self.m_rewardPerStep = rewardPerStep

        self.m_states = []
        self.m_statesMap = []

        self.m_actions = [ ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP ]
        self.m_actionMap = { ACTION_UP      : np.array( [ -1, 0 ], dtype = np.int32 ),
                             ACTION_DOWN    : np.array( [ 1, 0 ], dtype = np.int32 ), 
                             ACTION_LEFT    : np.array( [ 0, -1 ], dtype = np.int32 ),
                             ACTION_RIGHT   : np.array( [ 0, 1 ], dtype = np.int32 ) }

        _success = self._buildFromLayout()

        if not _success :
            self.m_gridLayout = DEFAULT_LAYOUT
            self._buildFromLayout()

        self.seed()
        self.reset()

    def reset( self ) :
        self.m_timestep = 0
        self.m_currentState = self._categoricalSample( self.m_isd )

        return self.m_currentState

    def seed( self, seed = None ) :
        self.m_randomState, self.m_seed = seeding.np_random( seed )
        return [self.m_seed]

    def step( self, action ) :
        # action sanity check
        assert self.action_space.contains( action ), "Wrong action given: %s" % str( action )

        # should follow action or slip?
        if self.m_randomState.random_sample() > self.m_noise :
            _actionToFollow = action
        else :
            _actionToFollow = self._pickSidewaysAction( action, self.m_randomState.random_sample() > 0.5 )

        # take a step in the simulation
        _snext = self._nextState( self.m_currentState, _actionToFollow )
        # grab the reward accordingly
        _reward = self._rewardFcn( self.m_currentState, _actionToFollow )
        # and check for termination
        _done = self._isTerminal( _snext )
        
        self.m_timestep += 1
        self.m_currentState = _snext

        # state sanity check
        assert self.observation_space.contains( _snext ), "Out of bounds: %s" % str( _snext )

        return [_snext, _reward, _done, {}]

    def render( self ) :
        plt.ion()
        plt.cla()

        _mat = np.zeros( ( self.m_rows, self.m_cols, 3 ) )

        for i in range( self.m_rows ) :
            for j in range( self.m_cols ) :
                _cellId = self.m_grid[i,j]
                _mat[i,j,0] = CELL_COLORS[_cellId][0]
                _mat[i,j,1] = CELL_COLORS[_cellId][1]
                _mat[i,j,2] = CELL_COLORS[_cellId][2]

        [_row, _col] = self._state2pos( self.m_currentState )
        # display the current state with another color
        _mat[ _row, _col, 0 ] = COLOR_CURRENT_POSITION[0]
        _mat[ _row, _col, 1 ] = COLOR_CURRENT_POSITION[1]
        _mat[ _row, _col, 2 ] = COLOR_CURRENT_POSITION[2]

        plt.imshow( _mat )
        plt.pause( 0.01 )
        
    @property
    def P( self ) :
        return self.m_transitionModel
    
    @property
    def nS( self ) :
        return self.observation_space.n
    
    @property
    def nA( self ) :
        return self.action_space.n
    
    @property
    def cols(self):
        return self.m_cols
    
    @property
    def rows(self):
        return self.m_rows
    

    ## Internal functionality ##################################################################################

    def _pos2state( self, row, col ) :
        return row * self.m_cols + col

    def _state2pos( self, index ) :
        _row = index // self.m_cols
        _col = index % self.m_cols
        return np.array( [ _row, _col ], dtype = np.int32 )

    def _buildFromLayout( self ) :
        # validate the grid layout
        self.m_rows = len( self.m_gridLayout )
        self.m_cols = len( self.m_gridLayout[0] )
        self.m_grid = np.zeros( ( self.m_rows, self.m_cols ), dtype = np.int32 )

        # build the grid representation
        for i in range( self.m_rows ) :
            _row = self.m_gridLayout[i]
            
            # validate the layout sizes
            if len( _row ) != self.m_cols :
                print( 'ERROR> GridworldInitialization: Gridworld layout mismatch' )
                return False

            for j in range( self.m_cols ) :
                # validate characters in the layout
                if _row[j] not in CELL_CHAR_MAP :
                    print( 'ERROR> GridworldInitialization: Gridworld layout wrong character: ', _row[j] )
                    return False
                # store the cell id
                self.m_grid[ i, j ] = CELL_CHAR_MAP[ _row[j] ]

        # configure underlying gym observation space
        self.observation_space = spaces.Discrete( self.m_rows * self.m_cols )
        # configure underlying gym action space
        self.action_space = spaces.Discrete( len( self.m_actions ) )

        self.m_states = range( self.m_rows * self.m_cols )
        self.m_statesMap = [ self._state2pos( s ) for s in self.m_states ]

        self.m_transitionModel = { s : { a : [] for a in self.m_actions } for s in self.m_states }

        # compute initial state probability distribution (where it can be at the beginning)
        self.m_isd = ( np.array( self.m_gridLayout ) == '.' ).astype( 'float64' ).ravel()
        print( 'isd: ', self.m_isd )
        self.m_isd /= self.m_isd.sum()

        for _row in range( self.m_rows ) :
            for _col in range( self.m_cols ) :
                for _action in self.m_actions :
                    # grab state id
                    _state = self._pos2state( _row, _col )
                    # if in a termination state, just set the next states to that state
                    if ( ( self.m_grid[_row, _col] == CELL_GOAL ) or
                         ( self.m_grid[_row, _col] == CELL_HOLE ) ) :
                        self.m_transitionModel[_state][_action].append( ( 1.0, _state, 0, True ) )
                    # if not, add all possible transitions using the transition function
                    else :
                        self.m_transitionModel[_state][_action].extend( self._transitionFcn( _state, _action ) )

        return True

    def _clampToBoundaries( self, row, col ) :
        row = max( min( row, self.m_rows - 1 ), 0 )
        col = max( min( col, self.m_cols - 1 ), 0 )

        return [row, col]

    def _checkBlockedCells( self, row, col, action ) :
        if self.m_grid[ row, col ] == CELL_BLOCKED :
            row = row - self.m_actionMap[ action ][0]
            col = col - self.m_actionMap[ action ][1]

        return [row, col]

    def _isTerminal( self, state ) :
        [ _row, _col ] = self._state2pos( state )
        # check if in hole or goal
        if ( ( self.m_grid[_row, _col] == CELL_GOAL ) or
             ( self.m_grid[_row, _col] == CELL_HOLE ) ) :
            return True

        return False

    def _pickSidewaysAction( self, action, sideChoice ) :
        if action == ACTION_UP or action == ACTION_DOWN :
            if sideChoice : 
                return ACTION_LEFT
            else : 
                return ACTION_RIGHT
        else :
            if sideChoice : 
                return ACTION_DOWN
            else : 
                return ACTION_UP

    def _transitionFcn( self, state, action ) :
        # transition pair 1 -> takes action with ( 1 - noise ) chance
        _tp1 = []
        _tp1.append( 1 - self.m_noise )
        _tp1.append( self._nextState( state, action ) )
        _tp1.append( self._rewardFcn( state, action ) )
        _tp1.append( self._isTerminal( state ) )
        # transition pair 2 -> takes side-action 1 with ( 0.5 * noise ) chance
        _tp2 = []
        _sideAction = self._pickSidewaysAction( action, True )
        _tp2.append( 0.5 * self.m_noise )
        _tp2.append( self._nextState( state, _sideAction ) )
        _tp2.append( self._rewardFcn( state, _sideAction ) )
        _tp2.append( self._isTerminal( state ) )
        # transition pair 3 -> takes side-action 2 with ( 0.5 * noise ) chance
        _tp3 = []
        _sideAction = self._pickSidewaysAction( action, False )
        _tp3.append( 0.5 * self.m_noise )
        _tp3.append( self._nextState( state, _sideAction ) )
        _tp3.append( self._rewardFcn( state, _sideAction ) )
        _tp3.append( self._isTerminal( state ) )

        return [ tuple( _tp1 ), tuple( _tp2 ), tuple( _tp3 ) ]

    def _rewardFcn( self, state, action ) :
        [_row, _col] = self._state2pos( state )
        # if at terminal state already, no reward
        if ( ( self.m_grid[_row, _col] == CELL_GOAL ) or
             ( self.m_grid[_row, _col] == CELL_HOLE ) ) :
            return 0.0
        else :
            _nstate = self._nextState( state, action )
            [ _nrow, _ncol ] = self._state2pos( _nstate )
            if self.m_grid[_nrow, _ncol] == CELL_GOAL :
                return self.m_rewardAtGoal
            elif self.m_grid[_nrow, _ncol] == CELL_HOLE :
                return self.m_rewardAtHole

        return self.m_rewardPerStep

    def _nextState( self, state, action ) :
        [_row, _col] = self._state2pos( state )

        _nrow = _row + self.m_actionMap[action][0]
        _ncol = _col + self.m_actionMap[action][1]

        [_nrow, _ncol] = self._clampToBoundaries( _nrow, _ncol )
        [_nrow, _ncol] = self._checkBlockedCells( _nrow, _ncol, action )

        return self._pos2state( _nrow, _ncol )

    def _categoricalSample( self, prob_n ):
        """
        Sample from categorical distribution
        Each row specifies class probabilities
        """
        prob_n = np.asarray( prob_n )
        csprob_n = np.cumsum( prob_n )
        return ( csprob_n > self.m_randomState.rand() ).argmax()

    ## #########################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseEnv, EnvInitializationException

CELL_EMPTY      = 0
CELL_BLOCKED    = 1
CELL_HOLE       = 2
CELL_GOAL       = 3
CELL_START      = 4

CELL_CHAR_MAP = { '.' : CELL_EMPTY, 
                  'B' : CELL_BLOCKED, 
                  'H' : CELL_HOLE, 
                  'G' : CELL_GOAL,
                  'S' : CELL_START }

CELL_COLORS = [ [ 0.0, 1.0, 0.0 ],
                [ 0.0, 0.0, 0.0 ],
                [ 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 1.0 ],
                [ 0.5, 0.5, 0.5 ] ]

COLOR_CURRENT_POSITION = [ 0.5, 0.5, 0.5 ]

ACTION_UP = 'Up'
ACTION_DOWN = 'Down'
ACTION_LEFT = 'Left'
ACTION_RIGHT = 'Right'

DEFAULT_LAYOUT = [ [ '.', '.', '.', 'G' ],
                   [ '.', 'B', '.', 'H' ],
                   [ '.', 'B', '.', '.' ],
                   [ 'S', '.', '.', '.' ] ]

class GridWorldEnv( BaseEnv ) :

    def __init__( self, gridLayout, 
                  gamma = 0.9, 
                  noise = 0.1,
                  rewardAtGoal = 100.0, 
                  rewardAtHole = -100.0,
                  rewardPerStep = -1.0 ) :
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
        self.m_gridLayout = gridLayout

        self.m_grid = None
        self.m_rows = 0
        self.m_cols = 0

        self.m_gamma = gamma
        self.m_noise = noise

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

    def step( self, state, action ) :
        # should follow action or slip?
        if np.random.random() > self.m_noise :
            _actionToFollow = action
        else :
            _actionToFollow = self._pickSidewaysAction( action, np.random.random() > 0.5 )

        _snext  = self._nextState( state, _actionToFollow )
        _reward = self._rewardFcn( state, _actionToFollow )
        _done   = self._isTerminal( _snext )

        _reward *= ( self.m_gamma ** self.m_timestep )
        
        self.m_timestep += 1

        return [_snext, _reward, _done]

    def render( self, currentState = None ) :
        plt.ion()
        plt.cla()

        _mat = np.zeros( ( self.m_rows, self.m_cols, 3 ) )

        for i in range( self.m_rows ) :
            for j in range( self.m_cols ) :
                _cellId = self.m_grid[i,j]
                _mat[i,j,0] = CELL_COLORS[_cellId][0]
                _mat[i,j,1] = CELL_COLORS[_cellId][1]
                _mat[i,j,2] = CELL_COLORS[_cellId][2]

        if currentState is not None :
            [_row, _col] = self.state2pos( currentState )
            # display the current state with another color
            _mat[ _row, _col, 0 ] = COLOR_CURRENT_POSITION[0]
            _mat[ _row, _col, 1 ] = COLOR_CURRENT_POSITION[1]
            _mat[ _row, _col, 2 ] = COLOR_CURRENT_POSITION[2]

        plt.imshow( _mat )

    def reset( self ) :
        self.m_timestep = 0

    def actions( self ) :
        return self.m_actions

    def actionsMap( self ) :
        return self.m_actionMap

    def states( self ) :
        return self.m_states

    def statesMap( self ) :
        return self.m_statesMap

    def pos2state( self, row, col ) :
        return row * self.m_cols + col

    def state2pos( self, index ) :
        _row = index // self.m_cols
        _col = index % self.m_cols
        return np.array( [ _row, _col ], dtype = np.int32 )

    def getTransitionModel( self ) :
        return self.m_transitionModel

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

        self.m_stateSpaceDim = ( self.m_rows, self.m_cols )
        self.m_actionSpaceDim = ( len( self.m_actions ), 1 )

        self.m_states = range( self.m_rows * self.m_cols )
        self.m_statesMap = [ self.state2pos( s ) for s in self.m_states ]

        self.m_transitionModel = { s : { a : [] for a in self.m_actions } for s in self.m_states }

        for _row in range( self.m_rows ) :
            for _col in range( self.m_cols ) :
                for _action in self.m_actions :
                    # grab state id
                    _state = self.pos2state( _row, _col )
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
        [ _row, _col ] = self.state2pos( state )
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
        _tp2.append( self._rewardFcn( state, action ) )
        _tp2.append( self._isTerminal( state ) )
        # transition pair 3 -> takes side-action 2 with ( 0.5 * noise ) chance
        _tp3 = []
        _sideAction = self._pickSidewaysAction( action, False )
        _tp3.append( 0.5 * self.m_noise )
        _tp3.append( self._nextState( state, _sideAction ) )
        _tp3.append( self._rewardFcn( state, action ) )
        _tp3.append( self._isTerminal( state ) )

        return [ tuple( _tp1 ), tuple( _tp2 ), tuple( _tp3 ) ]

    def _rewardFcn( self, state, action ) :
        [_row, _col] = self.state2pos( state )
        # if at terminal state already, no reward
        if ( ( self.m_grid[_row, _col] == CELL_GOAL ) or
             ( self.m_grid[_row, _col] == CELL_HOLE ) ) :
            return 0.0
        else :
            _nstate = self._nextState( state, action )
            [ _nrow, _ncol ] = self.state2pos( _nstate )
            if self.m_grid[_nrow, _ncol] == CELL_GOAL :
                return self.m_rewardAtGoal
            elif self.m_grid[_nrow, _ncol] == CELL_HOLE :
                return self.m_rewardAtHole

        return self.m_rewardPerStep

    def _nextState( self, state, action ) :
        [_row, _col] = self.state2pos( state )

        _nrow = _row + self.m_actionMap[action][0]
        _ncol = _col + self.m_actionMap[action][1]

        [_nrow, _ncol] = self._clampToBoundaries( _nrow, _ncol )
        [_nrow, _ncol] = self._checkBlockedCells( _nrow, _ncol, action )

        return self.pos2state( _nrow, _ncol )

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseEnv, EnvTest, EnvInitializationException

CELL_EMPTY = 0
CELL_BLOCKED = 1
CELL_HOLE = 2
CELL_GOAL = 3

CELL_CHAR_MAP = { '.' : CELL_EMPTY, 
                  'B' : CELL_BLOCKED, 
                  'H' : CELL_HOLE, 
                  'G' : CELL_GOAL }

CELL_COLORS = [ [ 0.0, 1.0, 0.0 ],
                [ 0.0, 0.0, 0.0 ],
                [ 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 1.0 ] ]

COLOR_CURRENT_POSITION = [ 0.5, 0.5, 0.5 ]

ACTION_UP = 'Up'
ACTION_DOWN = 'Down'
ACTION_LEFT = 'Left'
ACTION_RIGHT = 'Right'



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

        self.m_actions = [ ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT ]
        self.m_actionMap = { ACTION_UP      : np.array( [ -1, 0 ], dtype = np.int32 ),
                             ACTION_DOWN    : np.array( [ 1, 0 ], dtype = np.int32 ), 
                             ACTION_LEFT    : np.array( [ 0, -1 ], dtype = np.int32 ),
                             ACTION_RIGHT   : np.array( [ 0, 1 ], dtype = np.int32 ) }

        self._buildFromLayout()

    def _buildFromLayout( self ) :
        # validate the grid layout
        try :
            self.m_rows = len( self.m_gridLayout )
            self.m_cols = len( self.m_gridLayout[0] )
            self.m_grid = np.zeros( ( self.m_rows, self.m_cols ), dtype = np.int32 )
        except :
            print( 'ERROR> Invalid grid-layout given' )
            raise EnvInitializationException( 'Wrong gridworld layout', 
                                              'GridworldInitialization' )

        # build the grid representation
        for i in range( self.m_rows ) :
            _row = self.m_gridLayout[i]
            
            if len( _row ) != self.m_cols :
                raise EnvInitializationException( 'Gridworld layout mismatch',
                                                  'GridworldInitialization' )

            for j in range( self.m_cols ) :
                if _row[j] not in CELL_CHAR_MAP :
                    raise EnvInitializationException( 'Gridworld layout wrong character',
                                                      'GridworldInitialization' )
                self.m_grid[ i, j ] = CELL_CHAR_MAP[ _row[j] ]

        self.m_stateSpaceDim = ( self.m_rows, self.m_cols )
        self.m_actionSpaceDim = ( len( self.m_actions ), 1 )

    def _clampToBoundaries( self, state ) :
        _statec = state.copy()

        _statec[0] = max( min( state[0], self.m_rows - 1 ), 0 )
        _statec[1] = max( min( state[1], self.m_cols - 1 ), 0 )

        return _statec

    def _checkBlockedCells( self, state, action ) :
        if self.m_grid[ state[0], state[1] ] == CELL_BLOCKED :
            return state - self.m_actionMap[ action ]
        else :
            return state

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

    def _transition( self, state, action ) :
        _followAction = np.random.random() > self.m_noise

        if np.random.random() > self.m_noise :
            _actionToFollow = action
        else :
            _actionToFollow = self._pickSidewaysAction( action, np.random.random() > 0.5 )

        _snext = self._clampToBoundaries( state + self.m_actionMap[ _actionToFollow ] )
        _snext = self._checkBlockedCells( _snext, _actionToFollow )

        _row = _snext[0]
        _col = _snext[1]

        _reward = 0.0
        _done = False

        if self.m_grid[_row,_col] == CELL_GOAL :
            _reward = self.m_rewardAtGoal
            _done = True
        elif self.m_grid[_row,_col] == CELL_HOLE :
            _reward = self.m_rewardAtHole
            _done = True
        else :
            _reward = self.m_rewardPerStep

        return _snext, _reward, _done

    def transitionModel( self, state, action ) :
        # transition pair 1 -> takes action with ( 1 - noise ) chance
        _tp1 = {}
        _tp1['tpState'] = self._clampToBoundaries( state + self.m_actionMap[ action ] )
        _tp1['tpProb'] = ( 1 - self.m_noise )
        # transition pair 2 -> takes side-action 1 with ( 0.5 * noise ) chance
        _tp2 = {}
        _sideAction = self._pickSidewaysAction( action, True )
        _tp2['tpState'] = self._clampToBoundaries( state + self.m_actionMap[ _sideAction ] )
        _tp2['tpProb'] = ( 0.5 * self.m_noise )
        # transition pair 3 -> takes side-action 2 with ( 0.5 * noise ) chance
        _tp3 = {}
        _sideAction = self._pickSidewaysAction( action, False )
        _tp3['tpState'] = self._clampToBoundaries( state + self.m_actionMap[ _sideAction ] )
        _tp3['tpProb'] = ( 0.5 * self.m_noise )

        return [ _tp1, _tp2, _tp3 ]

    def reset( self ) :
        self.m_timestep = 0

    def step( self, state, action ) :
        _snext, _reward, _done = self._transition( state, action )
        _reward *= ( self.m_gamma ** self.m_timestep )
        
        self.m_timestep += 1

        return _snext, _reward, _done

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
            # display the current state with another color
            _mat[ currentState[0], currentState[1], 0 ] = COLOR_CURRENT_POSITION[0]
            _mat[ currentState[0], currentState[1], 1 ] = COLOR_CURRENT_POSITION[1]
            _mat[ currentState[0], currentState[1], 2 ] = COLOR_CURRENT_POSITION[2]

        plt.imshow( _mat )

    def actions( self ) :
        return self.m_actions

class GridWorldEnvInitializationTest( EnvTest ) :

    def __init__( self, nameid ) :
        super( GridWorldEnvInitializationTest, self ).__init__( nameid )
    
    def run( self ) :
        # test gridlayout
        _grid = [ [ '.', '.', '.', 'G' ],
                  [ '.', 'B', '.', 'H' ],
                  [ '.', 'B', '.', '.' ],
                  [ '.', '.', '.', '.' ] ]

        # test gridworld
        _env = GridWorldEnv( _grid )

class GridWorldEnvTransitionsTest( EnvTest ) :

    def __init__( self, nameid ) :
        super( GridWorldEnvTransitionsTest, self ).__init__( nameid )

    def run( self ) :
        # test gridlayout
        _grid = [ [ '.', '.', '.', 'G' ],
                  [ '.', 'B', '.', 'H' ],
                  [ '.', 'B', '.', '.' ],
                  [ '.', '.', '.', '.' ] ]

        # test gridworld
        _env = GridWorldEnv( _grid )
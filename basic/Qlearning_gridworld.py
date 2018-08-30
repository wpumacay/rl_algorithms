
import sys
import numpy as np
import matplotlib.pyplot as plt

CELL_CHAR_MAP = { '.' : 0, 'B' : 1, 'H' : 2, 'G' : 3 }
CELL_COLORS = [ [ 0.0, 1.0, 0.0 ],
                [ 0.0, 0.0, 0.0 ],
                [ 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 1.0 ] ]

CELL_EMPTY = 0
CELL_BLOCKED = 1
CELL_HOLE = 2
CELL_GOAL = 3

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4

class GridWorld( object ) :

    def __init__( self, grid, noise = 0.2 ) :
        # make plotting interactive
        plt.ion()
        # internals
        self.m_rows = 0
        self.m_cols = 0
        self.m_grid = None
        self.m_states = None
        self.m_noise = noise

        # discount factor
        self.m_gamma = 0.9
        self.m_t = 0

        # action space
        self.m_actionMap = np.array( [ [ 0, 1 ],
                                       [ 0, -1 ],
                                       [ -1, 0 ],
                                       [ 1, 0 ] ] )
        self.m_actions = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

        # initialize
        self._buildWorld( grid )

    def states( self ) :
        return self.m_states

    def actions( self ) :
        return self.m_actions

    def numRows( self ) :
        return self.m_rows

    def numCols( self ) :
        return self.m_cols

    def grid( self ) :
        return self.m_grid

    def noise( self ) :
        return self.m_noise

    def gamma( self ) :
        return self.m_gamma

    def reward( self, x ) :
        _row = x[0,1]
        _col = x[0,0]
        # check state
        if self.m_grid[_row][_col] == CELL_GOAL :
            return True, 100.0 * ( self.m_gamma ** self.m_t )
        elif self.m_grid[_row][_col] == CELL_HOLE :
            return True, -100.0
        else :
            return False, -1.0

    def reset( self ) :
        self.m_t = 0

    def _buildWorld( self, grid ) :
        try :
            self.m_rows = len( grid )
            self.m_cols = len( grid[0] )
            self.m_grid = np.zeros( ( self.m_rows, self.m_cols ), dtype = 'int32' )
            self.m_states = []
        except :
            print( 'An invalid grid was given' )
            sys.exit( -1 )
        # check that the grid is valid
        for i in range( self.m_rows ) :
            _row = grid[i]
            # sanity check
            assert ( len( _row ) == self.m_cols ), 'Invalid grid - dimension mismatch'
            for j in range( self.m_cols ) :
                assert ( _row[j] in CELL_CHAR_MAP ), 'Invalid grid - wrong character'
                self.m_grid[ i, j ] = CELL_CHAR_MAP[ _row[j] ]
                self.m_states.append( np.array( [ [ i, j ] ] ) )

    def _moveSideways( self, x, u, mdir = True ) :
            if u == ACTION_DOWN or u == ACTION_UP :
                if mdir :
                    return x + self.m_actionMap[ACTION_LEFT]
                else :
                    return x + self.m_actionMap[ACTION_RIGHT]
            else :
                if mdir :
                    return x + self.m_actionMap[ACTION_DOWN]
                else :
                    return x + self.m_actionMap[ACTION_UP]            


    def transitionModel( self, x, u ) :
        # transition pair 1 -> takes action with ( 1 - noise ) chance
        _tp1 = {}
        _tp1['tpState'] = self._clampToBoundaries( x + self.m_actionMap[u] )
        _tp1['tpProb'] = ( 1 - self.m_noise )
        # transition pair 2 -> takes side-action 1 with ( 0.5 * noise ) chance
        _tp2 = {}
        _tp2['tpState'] = self._clampToBoundaries( self._moveSideways( x, u, True ) )
        _tp2['tpProb'] = ( 0.5 * self.m_noise )
        # transition pair 3 -> takes side-action 2 with ( 0.5 * noise ) chance
        _tp3 = {}
        _tp3['tpState'] = self._clampToBoundaries( self._moveSideways( x, u, False ) )
        _tp3['tpProb'] = ( 0.5 * self.m_noise )

        return [ _tp1, _tp2, _tp3 ]

    def _clampToBoundaries( self, x ) :
        _xc = x.copy()
        _xc[0,0] = max( min( x[0][0], self.m_cols - 1 ), 0 )
        _xc[0,1] = max( min( x[0][1], self.m_rows - 1 ), 0 )

        return _xc

    def _transition( self, x, u, forceDeterministic ) :
        if forceDeterministic or ( np.random.random() > self.m_noise ) :
            # act normally
            return x + self.m_actionMap[u]
        else :
            # move sideways due to noise
            return self._moveSideways( x, u, np.random.random() > 0.5 )

    def step( self, x, u, forceDeterministic = False ) :
        # validate action
        assert ( 0 <= u and u < NUM_ACTIONS ), 'Invalid action requested'
        # validate state
        x = x.reshape( ( 1, 2 ) )
        
        # make step
        _xnext = self._transition( x, u, forceDeterministic )
        _xnext = self._clampToBoundaries( _xnext )
        # if blocked, just stay where you were
        if self.m_grid[ _xnext[0,1], _xnext[0,0] ] == CELL_BLOCKED :
            _xnext = x.copy()
        # collect reward for transition
        _done, _reward = self.reward( _xnext )
        # increase tick count
        self.m_t += 1
        
        return _xnext, _reward, _done

    def render( self ) :
        _mat = np.zeros( ( self.m_rows, self.m_cols, 3 ) )
        for i in range( self.m_rows ) :
            for j in range( self.m_cols ) :
                _cellId = self.m_grid[i][j]
                _mat[i,j,0] = CELL_COLORS[_cellId][0]
                _mat[i,j,1] = CELL_COLORS[_cellId][1]
                _mat[i,j,2] = CELL_COLORS[_cellId][2]
        plt.imshow( _mat )
        plt.show()

class QValueIterationAgent( object ) :

    def __init__( self, world ) :
        self.m_x = np.zeros( ( 1, 2 ) )
        self.m_world = world
        self.m_qtable = np.zeros( ( world.numRows(), world.numCols(), NUM_ACTIONS ) )

    # TODO: Change nomenclature from control type to rl type

    def train( self ) :
        _nIters = 10
        _states = self.m_world.states()
        _actions = self.m_world.actions()
        _gamma = self.m_world.gamma()
        for i in range( _nIters ) :
            for _s in _states :
                for _a in _actions :
                    _sdist = self.m_world.transitionModel( _s, _a )
                    for _tp in _sdist :
                        _newS = _tp['tpState']
                        _prob = _tp['tpProb']
                        _done, _reward = self.m_world.reward( _newS )
                        # bellman backup
                        _row, _col = _s[0,0], _s[0,1]
                        _nrow, _ncol = _newS[0,0], _newS[0,1]
                        self.m_qtable[_row,_col,_a] += _prob * ( _reward + 
                                                                 _gamma * np.max( self.m_qtable[_nrow,_ncol,:] ) )


class QlearningAgent( object ) :

    def __init__( self, world ) :
        # internals
        self.m_x = np.zeros( ( 1, 2 ) )
        self.m_world = world
        self.m_qTable = np.zeros( ( world.numRows(), world.numCols(), NUM_ACTIONS ) )
        # hyperparameters
        self.m_alpha = 0.1
        self.m_epsilon = 1.0

    def train( self, nEpisodes = 1000 ) :
        for e in range( nEpisodes ) :
            pass



    def policy( self, x ) :
        _row = x[0,0]
        _col = x[0,1]

        _action = np.argmax( self.m_qTable[ _row, _col ] )
        return _action
    

_grid = [ [ '.', '.', '.', 'G' ],
          [ '.', 'B', '.', 'H' ],
          [ '.', 'B', '.', '.' ],
          [ '.', '.', '.', '.' ] ]
_world = GridWorld( _grid )

_world.render()

_qvalueAgent = QValueIterationAgent( _world )
_qvalueAgent.train()
print( 'qtable' )
print( _qvalueAgent.m_qtable )

_key = input( 'press any key to exit' )
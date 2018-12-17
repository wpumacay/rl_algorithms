
import sys
sys.path.insert( 0, '..' )

import numpy as np
from envs.gridworld import GridWorldEnv


def test_env_transition() :
    # test gridlayout
    _grid = [ [ '.', '.', '.', 'G' ],
              [ '.', 'B', '.', 'H' ],
              [ '.', 'B', '.', '.' ],
              [ '.', '.', '.', '.' ] ]
    # test gridworld
    _env = GridWorldEnv( _grid, noise = 0.0 )
    _actions = _env.actions()
    # initial state
    _state = _env.pos2state( 0, 0 )

    _state1, _, _ = _env.step( _state, _actions[0] ) # left action
    _state2, _, _ = _env.step( _state, _actions[1] ) # down action
    _state3, _, _ = _env.step( _state, _actions[2] ) # right action
    _state4, _, _ = _env.step( _state, _actions[3] ) # up action

    print( "_state1: ", _state1 )
    print( "_state2: ", _state2 )
    print( "_state3: ", _state3 )
    print( "_state4: ", _state4 )

    assert np.all( _env.state2pos( _state1 ) == np.array( [ 0, 0 ] ) ),  'T1: Left action not pass'
    assert np.all( _env.state2pos( _state2 ) == np.array( [ 1, 0 ] ) ),  'T2: Down action not pass'
    assert np.all( _env.state2pos( _state3 ) == np.array( [ 0, 1 ] ) ),  'T3: Right action not pass'
    assert np.all( _env.state2pos( _state4 ) == np.array( [ 0, 0 ] ) ),  'T4: Up action not pass'

    print( 'ALL TESTS PASS' )

test_env_transition()
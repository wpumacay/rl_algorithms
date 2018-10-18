
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
    _state = np.array( [ 0, 0 ], dtype = np.int32 )

    _state1, _, _ = _env.step( _state, _actions[0] ) # up action
    _state2, _, _ = _env.step( _state, _actions[1] ) # down action
    _state3, _, _ = _env.step( _state, _actions[2] ) # left action
    _state4, _, _ = _env.step( _state, _actions[3] ) # right action

    assert np.all( _state1 == np.array( [ 0, 0 ] ) ),  'T1: Up action not pass'
    assert np.all( _state2 == np.array( [ 1, 0 ] ) ),  'T1: Down action not pass'
    assert np.all( _state3 == np.array( [ 0, 0 ] ) ),  'T1: Left action not pass'
    assert np.all( _state4 == np.array( [ 0, 1 ] ) ),  'T1: Right action not pass'


test_env_transition()
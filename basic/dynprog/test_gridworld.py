
import numpy as np
import matplotlib.pyplot as plt

from envs.gridworld import GridWorldEnv

_grid = [ [ '.', '.', '.', 'G' ],
          [ '.', 'B', '.', 'H' ],
          [ '.', 'B', '.', '.' ],
          [ '.', '.', '.', '.' ] ]
_env = GridWorldEnv( _grid, noise = 0.0 )
_actions = _env.actions()

## # print transition model
## print( _env.getTransitionModel() )

# test transition
_state = _env.pos2state( 0, 1 )
print( 'state-1: ', _state )
_state, _reward, _done = _env.step( _state, _actions[1] )
print( 'state-2: ', _state )

# test rendering
_env.render( currentState = _env.pos2state( 1, 0 ) )

_ = input( 'Press ENTER to continue ...' )

_state = _env.pos2state( 0, 0 )
_env.render( currentState = _state )
plt.pause( 1 )

_return = 0.0

for i in range( 10 ) :

    _state, _reward, _done = _env.step( _state, _actions[1] )
    _return += _reward
    print( 'state-', i, ': ', _state, ', reward: ', _reward )

    _env.render( currentState = _state )

    if _done :
        break

    plt.pause( 1 )

print( 'final state: ', _state )
print( 'final return: ', _return )

_ = input( 'Press ENTER to continue ...' )
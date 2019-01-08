
import time
from gridworld import *

_env = GridWorldEnv()
_state = _env.reset()

print( 'nS: ', _env.nS )
print( 'nA: ', _env.nA )
print( 'P: ', _env.P )

print( 'P[0]: ', _env.P[0] )
print( 'P[0][0]: ', _env.P[0][0] )
print( 'P[15][0]: ', _env.P[14][0] )

for _ in range( 1000 ) :

    _action = _env.action_space.sample()

    print( 'cState: ', _state )
    print( 'cAction: ', _action )

    _state, _reward, _done, _ = _env.step( _action )

    print( 'nState: ', _state )
    print( 'reward: ', _reward )
    print( 'done: ', _done )

    if _done :
        break

    _env.render()
    time.sleep( 0.5 )
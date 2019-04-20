
from gridworld import *

MAX_STEPS_PER_EPISODE = 1000

_env = GridWorldEnv( DRLBOOTCAMP_CLIFF_LAYOUT,
                     noise = 0.0,
                     rewardAtGoal = 10.0, 
                     rewardAtHole = -10.0,
                     rewardPerStep = 0.0,
                     renderInteractive = True )


_state = _env.reset()
_steps = 0

while True :

    if _env.userRequestFinish :
        break

    _action = _env.getUserAction()

    if _action == -1 :
        _env.render()
        continue

    _snext, _reward, _done, _ = _env.step( _action )
    _env.render()

    print( '**************************' )
    print( 's(t): ', _state )
    print( 'a(t): ', _action )
    print( 's(t+1): ', _snext )
    print( 'r(t+1): ', _reward )
    print( '**************************' )

    if _done :
        break

    if _steps >= MAX_STEPS_PER_EPISODE :
        break

    _steps += 1
    _state = _snext

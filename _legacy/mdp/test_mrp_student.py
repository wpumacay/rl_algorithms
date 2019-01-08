
from processes.mrp import MSimpleMRP, MSimpleMRPstate

# testing the markov process functionality
_stateC1        = MSimpleMRPstate( 'Class1' )
_stateC2        = MSimpleMRPstate( 'Class2' )
_stateC3        = MSimpleMRPstate( 'Class3' )
_statePass      = MSimpleMRPstate( 'Pass' )
_statePub       = MSimpleMRPstate( 'Pub' )
_stateFacebook  = MSimpleMRPstate( 'Facebook' )
_stateSleep     = MSimpleMRPstate( 'Sleep', terminal = True )

_stateC1.connections        = [ ( _stateC2, 0.5, -2.0 ), ( _stateFacebook, 0.5, -2.0 ) ]
_stateC2.connections        = [ ( _stateC3, 0.8, -2.0 ), ( _stateSleep, 0.2, -2.0 ) ]
_stateC3.connections        = [ ( _statePass, 0.6, -2.0 ), ( _statePub, 0.4, -2.0 ) ]
_statePass.connections      = [ ( _stateSleep, 1.0, 10.0 ) ]
_statePub.connections       = [ ( _stateC1, 0.2, 1.0 ), ( _stateC2, 0.4, 1.0 ), ( _stateC3, 0.4, 1.0 ) ]
_stateFacebook.connections  = [ ( _stateC1, 0.1, -1.0 ), ( _stateFacebook, 0.9, -1.0 ) ]
_stateSleep.connections     = [ ( _stateSleep, 1.0, 0.0 ) ]

_mrprocess = MSimpleMRP( { _stateC1.name : _stateC1,
                           _stateC2.name : _stateC2,
                           _stateC3.name : _stateC3,
                           _statePass.name : _statePass,
                           _statePub.name : _statePub,
                           _stateFacebook.name : _stateFacebook,
                           _stateSleep.name : _stateSleep } )

print( 'Transition Matrix:' )
print( _mrprocess.transitionMatrix() )

print( 'Reward Vector:' )
print( _mrprocess.rewardVector() )

_episode = _mrprocess.run( _stateC1 )

print( 'Episode: ' )
print( _episode )

_sdist = _mrprocess.runChain( [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0 ], 1000 )

print( 'Final Distribution: ' )
print( _sdist )

print( 'Value function: ' )
print( _mrprocess.computeValueFunction() )
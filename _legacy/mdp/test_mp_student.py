
from processes.mp import MSimpleMP, MSimpleMPstate

# testing the markov process functionality
_stateC1        = MSimpleMPstate( 'Class1' )
_stateC2        = MSimpleMPstate( 'Class2' )
_stateC3        = MSimpleMPstate( 'Class3' )
_statePass      = MSimpleMPstate( 'Pass' )
_statePub       = MSimpleMPstate( 'Pub' )
_stateFacebook  = MSimpleMPstate( 'Facebook' )
_stateSleep     = MSimpleMPstate( 'Sleep', terminal = True )

_stateC1.connections        = [ ( _stateC2, 0.5 ), ( _stateFacebook, 0.5 ) ]
_stateC2.connections        = [ ( _stateC3, 0.8 ), ( _stateSleep, 0.2 ) ]
_stateC3.connections        = [ ( _statePass, 0.6 ), ( _statePub, 0.4 ) ]
_statePass.connections      = [ ( _stateSleep, 1.0 ) ]
_statePub.connections       = [ ( _stateC1, 0.2 ), ( _stateC2, 0.4 ), ( _stateC3, 0.4 ) ]
_stateFacebook.connections  = [ ( _stateC1, 0.1 ), ( _stateFacebook, 0.9 ) ]
_stateSleep.connections     = [ ( _stateSleep, 1.0 ) ]

_mprocess = MSimpleMP( { _stateC1.name : _stateC1,
                         _stateC2.name : _stateC2,
                         _stateC3.name : _stateC3,
                         _statePass.name : _statePass,
                         _statePub.name : _statePub,
                         _stateFacebook.name : _stateFacebook,
                         _stateSleep.name : _stateSleep } )

print( 'Transition Matrix:' )
print( _mprocess.transitionMatrix() )

_episode = _mprocess.run( _stateC1 )

print( 'Episode: ' )
print( _episode )

_sdist = _mprocess.runChain( [ 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0 ], 1000 )

print( 'Final Distribution: ' )
print( _sdist )
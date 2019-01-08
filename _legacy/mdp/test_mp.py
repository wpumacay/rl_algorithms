
from processes.mp import MSimpleMP, MSimpleMPstate

# testing the markov process functionality
_stateA = MSimpleMPstate( 'A' )
_stateB = MSimpleMPstate( 'B' )

_stateA.connections = [ ( _stateA, 0.9 ), ( _stateB, 0.1 ) ]
_stateB.connections = [ ( _stateA, 0.7 ), ( _stateB, 0.3 ) ]

_mprocess = MSimpleMP( { _stateA.name : _stateA,
                         _stateB.name : _stateB } )

print( 'Transition Matrix:' )
print( _mprocess.transitionMatrix() )

_episode = _mprocess.run( _stateA )

print( 'Episode: ' )
print( _episode )

_sdist = _mprocess.runChain( [ 0.2, 0.8 ] )

print( 'Final Distribution: ' )
print( _sdist )
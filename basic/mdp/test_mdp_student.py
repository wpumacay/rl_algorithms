
import random
from processes.mdp import MSimpleMDP, MSimpleMDPstate, MSimplePolicyInterface

# testing the markov process functionality
_stateC1        = MSimpleMDPstate( 'Class1' )
_stateC2        = MSimpleMDPstate( 'Class2' )
_stateC3        = MSimpleMDPstate( 'Class3' )
_statePass      = MSimpleMDPstate( 'Pass' )
_statePub       = MSimpleMDPstate( 'Pub', chanceNode = True )
_stateFacebook  = MSimpleMDPstate( 'Facebook' )
_stateSleep     = MSimpleMDPstate( 'Sleep', terminal = True )

_stateC1.connections        = [ ( _stateC2, 0.5, -2.0 ), ( _stateFacebook, 0.5, -2.0 ) ]
_stateC2.connections        = [ ( _stateC3, 0.8, -2.0 ), ( _stateSleep, 0.2, -2.0 ) ]
_stateC3.connections        = [ ( _statePass, 0.6, -2.0 ), ( _statePub, 0.4, -2.0 ) ]
_statePass.connections      = [ ( _stateSleep, 1.0, 10.0 ) ]
_statePub.connections       = [ ( _stateC1, 0.2, 1.0 ), ( _stateC2, 0.4, 1.0 ), ( _stateC3, 0.4, 1.0 ) ]
_stateFacebook.connections  = [ ( _stateC1, 0.1, -1.0 ), ( _stateFacebook, 0.9, -1.0 ) ]
_stateSleep.connections     = [ ( _stateSleep, 1.0, 0.0 ) ]

_states = { _stateC1.name : _stateC1,
            _stateC2.name : _stateC2,
            _stateC3.name : _stateC3,
            _statePass.name : _statePass,
            _statePub.name : _statePub,
            _stateFacebook.name : _stateFacebook,
            _stateSleep.name : _stateSleep }

_actions = { 'Study' : { _stateC1.name : _stateC2, 
                         _stateC2.name : _stateC3,
                         _stateC3.name : _statePass },
             'Facebook' : { _stateC1.name : _stateFacebook },
             'Quit' : { _stateFacebook.name : _stateC1 },
             'Pub' : { _stateC3.name : _statePub },
             'Sleep' : { _stateC2.name : _stateSleep,
                         _statePass.name : _stateSleep,
                         _stateSleep.name : _stateSleep } }

_mdprocess = MSimpleMDP( _states, _actions )

class RandomPolicy( MSimplePolicyInterface ) :

    def __init__( self, mdp ) :
        super( RandomPolicy, self ).__init__( 'Random Policy', mdp )

    def getAction( self, state ) :
        # get valid actions from state
        _validActions = []
        _mdpActions = self.m_mdp.actions()
        for _aname in _mdpActions :
            if state.name in _mdpActions[ _aname ] :
                _validActions.append( _aname )
        
        if len( _validActions ) < 1 :
            print( 'There is no action from state: ', state.name, ' .',
                   'Something is wrong with the MDP definition' )
            return None

        return _mdpActions[ random.choice( _validActions ) ]

_policy = RandomPolicy( _mdprocess )
_episode = _mdprocess.run( _stateC1, _policy )

print( 'Episode: ' )
print( _episode )
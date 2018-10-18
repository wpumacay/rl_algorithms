
import numpy as np


class BaseEnv ( object ) :

    def __init__( self ) :
        super( BaseEnv, self ).__init__()

        self.m_stateSpaceDim = 0
        self.m_actionSpaceDim = 0
        self.m_timestep = 0

    def step( self, action ) :
        pass

    def actionSpaceDim( self ) : 
        return self.m_actionSpaceDim

    def stateSpaceDim( self ) :
        return self.m_stateSpaceDim


class EnvInitializationException( Exception ) :

    def __init__( self, message, info ) :
        super( EnvInitializationException, self ).__init__( message )
        # extra information about the exception
        self.info = info


class EnvTest( object ) :

    def __init__( self, nameid ) :
        super( EnvTest, self ).__init__()

        self.m_nameid = nameid

    def nameid( self ) :
        return self.m_nameid

    def run( self ) : 
        return False

    
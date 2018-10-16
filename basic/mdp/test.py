
import numpy as np
import matplotlib.pyplot as plt


# ###########################################################

### drawing functionality simple-mdp

class MDrawable( object ) :

    def __init__( self, name ) :

        # public functionality #########
        self.name = name

        # ##############################

        # private functionality ########
        self.m_patch = None

        # ##############################

    def patch( self ):
        return self.m_patch



class MCircle( MDrawable ) :

    def __init__( self, name ) :
        super( MCircle, self ).__init__( name )


class MRectangle( MDrawable ) :

    def __init__( self, name ) :
        super( MRectangle, self ).__init__( name )



def createCircle( name, radius ) :
    return None

def createRectangle( name, width, height ) :
    return None




class MCanvas( object ) :

    def __init__( self ) :
        self.m_drawables = {}
        # initialize matplotlib resources
        # .... _fig, _ax = plt.
    
    def getDrawableByName( self, name ) :
        if name not in self.m_drawables :
            print( 'WARNING> Drawable with name: ', name, ' does not exist' )
            return None

        return self.m_drawables[ name ]

    def draw( self ):
        pass


class MApp ( object ) :

    def __init__( self ) :
        self.m_canvas = None
        self.m_eventHandler = None


def createAppFromMP():
    pass

def createAppFromMRP():
    pass

def createAppFromMDP():
    pass

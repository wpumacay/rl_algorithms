
# Helper functions adapted from the discretization part of the udacity deeprlnd:
# https://github.com/udacity/deep-reinforcement-learning/tree/master/discretization

import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

# Discretization utils #########################################################

def createTilingGrid( sLow, sHigh, nBins, offsets ) :
    """
        Creates a tiling of a state space given ...
        the min-max ranges for each its dimension.

        Parameters
        ----------
        sLow : array of floats
            low limits of the state space (each dimension)

        sHigh : array of floats
            high limits of the state space (each dimension)

        nBins : array of ints
            number of bins to partition each dimension

        offsets : array of floats
            grid offsets applied for each dimension

        Returns
        -------
        out : array of partitions for each dimension
            A grid (as an array of arrays) representing the tiling

    """

    _ndims = len( nBins )
    _grid = []

    for dim in range ( _ndims ) :
        _gdimension = np.linspace( sLow[dim], sHigh[dim], nBins[dim] + 1 ) + offsets[dim]
        _grid.append( _gdimension[1:-1] )

    return _grid

def createTilings( sLow, sHigh, tilingsSpecs ) :
    """
        Creates a set of tilings for the given specs, which ...
        contain bining sizes and offsets for each tiling.
        
        Parameters
        ----------
        sLow : array of floats
            low limits of the state space (each dimension)

        sHigh : array of floats
            high limits of the state space (each dimension)

        tilingSpecs : array of spec. tuples
            specs. for each tiling [((nbin1,nbin2,...),(off1,off2,...)), ...]

        Returns
        -------
        out : array of tilings
            An array of grid tilings, each from the given specs
    """

    return [ createTilingGrid( sLow, sHigh, tSpec[0], tSpec[1] ) for tSpec in tilingsSpecs ]

def getEncoding( state, tiling ) :
    """
        Returns the encoding of the given state in the ...
        given tiling of the state space.

        Parameters
        ----------
        state : array of floats
            Array representing the state to encode

        tiling : array of binnings(arrays as well)
            A tiling of the state space

        Returns
        -------
        out : array of ints
            Array representing the encoding of this state

    """

    return tuple( np.digitize( stateDim, binDim ) for stateDim, binDim in zip( state, tiling ) )

def getTilesEncodings( state, tilings ) :
    """
        Returns the encodings of the state in each tiling from the given tilings

        Parameters
        ----------
        state : array of floats
            Array representing the state to encode

        tilings : array of tilings(arrays as well)
            Tilings in which the state space was partitioned

        Returns
        -------
        out : array of arrays(encodings)
            Array with each encoding of the state in each tiling
    """

    return [ getEncoding( state, _tiling ) for _tiling in tilings ]


# Visualization utils ##########################################################

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

def drawTilings( tilings, dimensions = (0, 1) ) :
    """
        Draws a set of tilings over a set of dimensions of the state space.
    
        Parameters
        ----------
        tilings : Array of tilings
            The tilings in which the state space was partitioned

        dimensions : tuple of ints
            The dimensions of the state space to be visualized

        Returns
        -------
        fig : plt.Figure
            Figure used for the plot

        axes : plt.Axes
            Axes of the figure
    """

    _lineColors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    _lineStyles = ['-', '--', ':']
    _legends = []

    _fig, _axes = plt.subplots( figsize = (10, 10) )

    _tilingsOverDim = np.array( tilings )[:,dimensions,...]

    for k, _grid in enumerate( _tilingsOverDim ) :
        _color = _lineColors[k % len( _lineColors )]
        _style = _lineStyles[k % len( _lineStyles )]

        for _x in _grid[0] :
            _l = _axes.axvline( x = _x, color = _color, linestyle = _style, label = k )
        for _y in _grid[1] :
            _l = _axes.axhline( y = _y, color = _color, linestyle = _style )

        _legends.append( _l )

    _axes.grid( 'off' )
    _axes.legend( _legends, 
                  ['Tiling #{}'.format(t) for t in range(len(_tilingsOverDim))],
                  facecolor='white',
                  framealpha = 0.9 )
    _axes.set_title( 'Tilings for dimensions %d, %d' % tuple( dimensions ) )

    return _fig, _axes


def drawEncodings( states, encodings, tilings, dimensions = (0, 1), low = None, high = None ) :
    """
        Draws the tilings of a state space along with ...
        the encodings of a given set of state, all over ...
        some given dimensions of the state space

        Parameters
        ----------
        states : Array of states(arrays of floats)
            Array of states from the state space

        encodings : Array of encodings
            Array of encodings of the given states onto a set of tilings

        tilings : Array of tilings
            The tilings in which the state space was partitioned

        dimensions : tuple of ints
            The dimensions of the state space to be visualized

        low : array of floats
            Low limits of the state space in the given dimensions

        high : array of floats
            High limits of the state space in the given dimensions
    """

    # easier indexing (numpy slicing)
    states = np.array( states )
    encodings = np.array( encodings )
    tilings = np.array( tilings )
    dimensions = np.array( dimensions, dtype = np.int64 )

    # Grab the tiling for only the dimensions given by the user
    _statesOverDim = states[:,dimensions]
    _encodingsOverDim = encodings[:,:,dimensions]
    _tilingsOverDim = tilings[:,dimensions,...]

    # draw tiling grids
    _fig, _axes = drawTilings( tilings, dimensions )

    # set x-y bounds of the grid
    if ( low  is not None ) and ( high is not None ) :
        _axes.set_xlim( low[0], high[0] )
        _axes.set_ylim( low[1], high[1] )
    else :
        # pre-render to get the actual limits
        _axes.plot( _statesOverDim[:,0], _statesOverDim[:,1], 'o', alpha = 0.0 )
        low = [ _axes.get_xlim()[0], _axes.get_ylim()[0] ]
        high = [ _axes.get_xlim()[1], _axes.get_ylim()[1] ]

    # Map each encoded sample (which is really a list of indices) to the corresponding tiles it belongs to
    _tilingsExtended = [ np.hstack( ( np.array( [low] ).T, _tiling, np.array( [high] ).T ) ) for _tiling in _tilingsOverDim ]  # add low and high ends
    _tileCenters = [ ( _tilingExtended[:, 1:] + _tilingExtended[:, :-1]) / 2 for _tilingExtended in _tilingsExtended ]  # compute center of each tile
    _tileToplefts = [ _tilingExtended[:, :-1] for _tilingExtended in _tilingsExtended ]  # compute topleft of each tile
    _tileBottomrights = [ _tilingExtended[:, 1:] for _tilingExtended in _tilingsExtended]  # compute bottomright of each tile

    _colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for _state, _encodings in zip( _statesOverDim, _encodingsOverDim ) :
        for i, _encoding in enumerate( _encodings ) :
            # Shade the entire tile with a rectangle
            _topleft = _tileToplefts[i][0][_encoding[0]], _tileToplefts[i][1][_encoding[1]]
            _bottomright = _tileBottomrights[i][0][_encoding[0]], _tileBottomrights[i][1][_encoding[1]]
            _axes.add_patch( Rectangle( _topleft, 
                                        _bottomright[0] - _topleft[0], 
                                        _bottomright[1] - _topleft[1],
                                        color = _colors[i], 
                                        alpha = 0.33 ) )

            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any( _state < _topleft ) or any( _state > _bottomright ):
                # So plot a point in the center of the tile and draw a connecting line
                _cx, _cy = _tileCenters[i][0][_encoding[0]], _tileCenters[i][1][_encoding[1]]
                _axes.add_line( Line2D( [_state[0], _cx], [_state[1], _cy], color = _colors[i] ) )
                _axes.plot( _cx, _cy, 's', color = _colors[i] )
    
    # Finally, plot original samples
    _axes.plot( _statesOverDim[:, 0], _statesOverDim[:, 1], 'o', color = 'r' )

    _axes.margins( x = 0, y = 0 )  # remove unnecessary margins
    _axes.set_title( "Tile-encoded samples" )

    set_trace()

    return _fig ,_axes


## Some tests for the tools ####################################################

def test_tiling_space_2d() :
    plt.ion()
    # Tiling specs: [(<bins>, <offsets>), ...]
    _tilingSpecs = [ ( ( 10, 10 ), ( -0.066, -0.33 ) ),
                     ( ( 10, 10 ), ( 0.0, 0.0 ) ),
                     ( ( 10, 10 ), ( 0.066, 0.33 ) ) ]
    _tilings = createTilings( [-1., -5.], [1., 5.], _tilingSpecs )

    ## # draw these tilings
    ## drawTilings( _tilings )

    # Test with some sample values
    _samples = [ (-1.2 , -5.1 ) ,
                 (-0.75,  3.25) ,
                 (-0.5 ,  0.0 ) ,
                 ( 0.25, -1.9 ) ,
                 ( 0.15, -1.75) ,
                 ( 0.75,  2.5 ) ,
                 ( 0.7 , -3.7 ) ,
                 ( 1.0 ,  5.0 ) ]
    _encodedSamples = [ getTilesEncodings( sample, _tilings ) for sample in _samples ]

    # draw the encodings
    drawEncodings( _samples, _encodedSamples, _tilings )

    # wait for user to terminate
    _ = input( 'Press any key to continue' )

if __name__ == '__main__' :
    test_tiling_space_2d()
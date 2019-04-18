
import itertools

import numpy as np
import matplotlib.pyplot as plt



def _plotStateTableInGrid( table, rows, cols, title = 'State table' ) : 

    plt.figure()
    plt.title( title )

    _matrix = np.zeros( ( rows, cols ), dtype=np.float32 )

    for _s in table.keys() :
        # convert state id to grid position
        _row = _s // cols
        _col = _s % cols
        # set the value V(s) to the matrix
        _matrix[_row,_col] = table[_s]

    plt.imshow( _matrix, 
                interpolation = 'nearest', 
                cmap = plt.cm.Blues )

    for i, j in itertools.product( range( _matrix.shape[0] ),
                                   range( _matrix.shape[1] ) ) :
        plt.text( j, i, '{0:.2f}'.format( _matrix[i, j] ),
                  horizontalalignment = 'center',
                  color = 'black' )

    plt.tight_layout()
    plt.show()

def plotVTableInGrid( vtable, rows, cols ) :
    _plotStateTableInGrid( vtable, rows, cols, 'V-value for each state V(s)' )

def plotVisitsInGrid( visitsTable, rows, cols ) :
    _plotStateTableInGrid( visitsTable, rows, cols, 'Number of visits for each state N(s)' )

def plotVisitsInHistogram( visitsTable, nStates ) :
    _centers = np.arange( 0, nStates )

    _hist = np.zeros( _centers.shape )

    for _s in visitsTable.keys() :
        _hist[_s] = visitsTable[_s]

    plt.figure()
    plt.title( 'Visits histogram' )
    plt.bar( _centers, _hist, align = 'center', width = 1 )
    plt.show()


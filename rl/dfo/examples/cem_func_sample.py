
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from IPython.core.debugger import set_trace

plt.ion()

N = 100                 # population size
K = 0.25                # top k% to be selected
N_ELITES = int( K * N ) # number of elites to pick
ALPHA = 0.8             # learning rate to use to update the dist. parameters
MAX_ITERS = 1000        # max number of iterations to run

X_MIN = -5.0
X_MAX = 5.0

## X_MIN = -500.0
## X_MAX = 500.0


def f( x ) :
    return 5. - np.sum( x * x, axis = 1 )
    ## return 418.9829 * 2 - np.sum( x * np.sin( np.sqrt( np.abs( x ) ) ) , axis = 1 )

def plotf() :
    _xx1 = np.arange( 2. * X_MIN, 2. * X_MAX, 0.01 )
    _xx2 = np.arange( 2. * X_MIN, 2. * X_MAX, 0.01 )

    ## _xx1 = np.arange( 2. * X_MIN, 2. * X_MAX, 0.5 )
    ## _xx2 = np.arange( 2. * X_MIN, 2. * X_MAX, 0.5 )

    _X1, _X2 = np.meshgrid( _xx1, _xx2 )
    _X = np.c_[ _X1.ravel(), _X2.ravel() ]
    _Z = f( _X ).reshape( _X1.shape )

    plt.contour( _X1, _X2, _Z )

plotf()
plt.grid( True )
plt.xlim( ( 2. * X_MIN, 2. * X_MAX ) )
plt.ylim( ( 2. * X_MIN, 2. * X_MAX ) )

x_mean  = X_MIN + ( X_MAX - X_MIN ) * np.random.random_sample( (2,) )
x_std   = 2. * np.ones( (2,) ) * np.abs( X_MAX - X_MIN ) / 2.
f_best = -np.inf

# just a patches.Circle to indicate the gaussian distribution
GAUSSIAN_DIST_FIG = patches.Circle( ( x_mean[0], x_mean[1] ), radius = np.sqrt( np.sum( x_std ** 2 ) ) )
plt.gca().add_patch( GAUSSIAN_DIST_FIG )

def plotDistribution( xmean, xstd ) :
    # just update the patch position and radius
    GAUSSIAN_DIST_FIG.center = ( xmean[0], xmean[1] )
    GAUSSIAN_DIST_FIG.radius = np.sqrt( np.sum( xstd ** 2 ) )

_plt_elites_handle = None

def plotElites( elites, pointsHandle ) :
    if not pointsHandle :
        pointsHandle = plt.plot( elites[:,0], elites[:,1], 'rx' )[0]
    else :
        pointsHandle.set_data( elites[:,0], elites[:,1] )

    return pointsHandle

for i in range( MAX_ITERS ) :
    # sample from the normal distribution
    _x_samples = x_mean + x_std * ( 2. * np.random.random( ( N, 2 ) ) - 1. )

    # evaluate all samples
    _fscores = f( _x_samples )

    # pick the top k%
    _elites_indices = np.argsort( _fscores )[-N_ELITES:]
    _x_elites = _x_samples[_elites_indices]
    _x_current_best = _x_samples[_elites_indices[-1]]
    _f_current_best = np.max( _fscores )

    # update current best
    f_best = max( f_best, _f_current_best )

    # fit a new gaussian distribution to the elites
    x_mean = ( 1. - ALPHA ) * x_mean + ALPHA * np.mean( _x_elites )
    x_std = ( 1. - ALPHA ) * x_std + ALPHA * np.std( _x_elites )
    x_std = np.clip( x_std, 1., X_MAX - X_MIN )
    # x_std = np.clip( x_std, 10., X_MAX - X_MIN )

    plotDistribution( x_mean, x_std )
    _plt_elites_handle = plotElites( _x_elites, _plt_elites_handle )

    print( 'x_best: ', _x_current_best, ' - f_best: ', f_best )
    plt.pause( 1. )

_ = input( 'Press ENTER to continue ...' )
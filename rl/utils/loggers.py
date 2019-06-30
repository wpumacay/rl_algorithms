
import os
import sys
import abc
import numpy as np
from tqdm import tqdm
from collections import deque


class Logger( abc.ABC ) :
    r"""Base class for all logger implementations

    This class provides a handy logger interface for the user to
    derive and implement the required logging-to-anything functionality
    (either logging to file, console, etc.) and in which format and flavour
    the user might require. The purpose of this base class is to reduce
    boiler plate code.

    """
    def __init__( self, config = {} ) :
        super( Logger, self ).__init__()

        self._config = config
        self._maxEpisodes = config.get( 'maxEpisodes', 1000 )
        self._logWindowSize = config.get( 'logWindowSize', 100 )


        self._maxScore = -np.inf
        self._maxAvgScore = -np.inf
        self._currentScore = 0.
        self._currentAvgScore = 0.
        self._scoresArray = []
        self._scoresAvgsArray = []
        self._scoresWindow = deque( maxlen = self._logWindowSize )

        self._maxNsteps = -np.inf
        self._maxAvgNsteps = -np.inf
        self._currentSteps = 0
        self._currentAvgNsteps = 0
        self._nstepsArray = []
        self._nstepsAvgsArray = []
        self._nstepsWindow = deque( maxlen = self._logWindowSize )

        self._nepisodes = 0


    def update( self, transition ) :
        r"""Updates the logger state from a step given by a transition tuple

        Args:
            transition (tuple): transition tuple in the form (s,a,r,s',done)

        """
        # unpack the transition
        _, _, _r, _, _ = transition

        self._currentScore += _r
        self._currentSteps += 1


    def onEndEpisode( self ) :
        r"""Updates the logger state when an episode finishes

        """

        self._nepisodes += 1

        self._maxScore = max( self._maxScore, self._currentScore )
        self._scoresArray.append( self._currentScore )
        self._scoresWindow.append( self._currentScore )

        self._maxNsteps = max( self._maxNsteps, self._currentSteps )
        self._nstepsArray.append( self._currentSteps )
        self._nstepsWindow.append( self._currentSteps )

        if self._nepisodes >= self._logWindowSize :
            self._currentAvgScore = np.mean( self._scoresWindow )
            self._currentAvgNsteps = np.mean( self._nstepsWindow )

            self._maxAvgScore = max( self._maxAvgScore, self._currentAvgScore )
            self._maxAvgNsteps = max( self._maxAvgNsteps, self._currentAvgNsteps )

            self._scoresAvgsArray.append( self._currentAvgScore )
            self._nstepsAvgsArray.append( self._currentAvgNsteps )

        self.log()

        # clear counters and accumulators for next iteration
        self._currentScore = 0.
        self._currentSteps = 0

    @abc.abstractmethod
    def log( self ) :
        r"""Logs the current state of the logger

        This is an abstract method, and should be implemented for the
        appropriate flavour of logger you are using.

        """
        pass


class LoggerTqdm( Logger ) :
    r"""A logger to console that used tqdm

    This class implements a simple tqdm logger, by updating the description
    of the progressbar used in console.

    """
    def __init__( self, config = {} ) :
        super( LoggerTqdm, self ).__init__( config )

        self._progressbar = tqdm( range( self._maxEpisodes ), desc = 'Running>', leave = True )


    def log( self ) :
        if self._nepisodes >= self._logWindowSize :
            # create the message we will be setting as description
            _messageStr = 'Running> Best=%.3f, Curr=%.3f, Best-avg=%.3f, Curr-avg=%.3f'
            # create the required information to be replaced in the description
            _messageParams = ( self._maxScore,
                               self._currentScore,
                               self._maxAvgScore, 
                               self._currentAvgScore )
        else :
            # create the message we will be setting as description
            _messageStr = 'Running> Curr=%.3f, BestScore=%.3f'
            # create the required information to be replaced in the description
            _messageParams = ( self._currentScore,
                               self._maxScore )

        self._progressbar.update() # increate counter by 1 (as not using it on loop)
        self._progressbar.set_description( _messageStr % _messageParams )
        self._progressbar.refresh() # refresh text and other stuff

class LoggerFile( Logger ) :
    r"""A logger that stores info in a file

    This class implements a simple file logger, which appends 
    new logs into a file

    """
    def __init__( self, config = {} ) :
        super( LoggerFile, self ).__init__( config )

        # file where to save the logs
        self._filename = config.get( 'filename', 'logs' ) + '.txt'

    def log( self ) :
        if self._nepisodes >= self._logWindowSize :
            # create the message we will be setting as description
            _messageStr = 'Running> Best=%.3f, Curr=%.3f, Best-avg=%.3f, Curr-avg=%.3f \n\r'
            # create the required information to be replaced in the description
            _messageParams = ( self._maxScore,
                               self._currentScore,
                               self._maxAvgScore, 
                               self._currentAvgScore )
        else :
            # create the message we will be setting as description
            _messageStr = 'Running> Curr=%.3f, BestScore=%.3f \n\r'
            # create the required information to be replaced in the description
            _messageParams = ( self._currentScore,
                               self._maxScore )

        with open( self._filename, 'a' ) as fhandle :
            fhandle.write( _messageStr % _messageParams )
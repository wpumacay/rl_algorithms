
import os
import sys
import abc
import random
import numpy as np
from collections import deque

from rl.utils.config import TrainerConfig
from rl.utils.loggers import LoggerTqdm

class Trainer( abc.ABC ) :
    r"""Base class for all trainers

    Trainers consist of object in charge of the training and testing 
    process. This provides a handy interface to handle your training
    loop. Some implementations of basic trainers are implemented below,
    but the user is advised to implement her|his own trainer according
    to its requirements. The main idea is to abstract the training|testing 
    loop boiler plate code away from the user such that she|he can focus
    only on the agent's implementation

    Args:
        trainerConfig (TrainerConfig): configuration for the trainer

    """
    def __init__( self, trainerConfig ) :
        super( Trainer, self ).__init__()

        # save the configuration dictionary for future reference
        self._config = trainerConfig

        # grab some information from the trainer configuration
        self._maxEpisodes       = self._config.maxEpisodes
        self._maxSteps          = self._config.maxStepsPerEpisode
        self._logWindowSize     = self._config.logWindowSize
        self._loggerType        = self._config.loggerType
        self._seed              = self._config.seed
        self._mode              = self._config.mode
        self._testOnceTrained   = self._config.testOnceTrained
        self._numTestEpisodes   = self._config.numTestEpisodes


    @abc.abstractmethod
    def init( self ) :
        r"""Abstract method for trainer initialization

        Derive and implement your own functionality for your trainer 
        initialization, according to your setup requirements

        """
        pass


    @abc.abstractmethod
    def train( self, args = {} ) :
        r"""Abstract method to start the training process

        Derive and implement your own functionality for your trainer 
        training process, according to your setup requirements

        """
        pass


    @abc.abstractmethod
    def test( self, args = {} ) :
        r"""Abstract method to start the testing process

        Derive and implement your own functionality for your trainer 
        testint process, according to your setup requirements

        """
        pass


    @property
    def seed( self ):
        return self._seed


    @property
    def mode( self ) :
        return self._mode


    @property
    def testOnceTrained( self ) :
        return self._testOnceTrained


class SimpleTrainer( Trainer ) :
    r"""Creates a simple single-agent single-environment non-parallel trainer
    
    This is the most basic trainer, derived from the Trainer interface. Use it
    for simple setups that require only one agent and one environment running
    in a single process (no parallelism).

    Args:
        trainerConfig (TrainerConfig): configuration for the trainer
        env (Environment): environment to run training on
        agent (Agent): agent to train

    """
    def __init__( self, trainerConfig, env, agent ) :
        super( SimpleTrainer, self ).__init__( trainerConfig )

        # single environment to be used for our setup
        self._env = env
        # single agent to be used for our setup
        self._agent = agent

        assert self._env is not None, 'ERROR> Could not create the environment'
        assert self._agent is not None, 'ERROR> Could not create the agent'

        # logger to be used for saving training information
        self._logger = None


    def init( self ) :
        self._env.seed( self._seed )
        # seed numpy and random-module random generators
        np.random.seed()
        random.seed()

        if self._loggerType == 'tqdm' :
            self._logger = LoggerTqdm( { 'maxEpisodes' : self._maxEpisodes,
                                         'logWindowSize' : self._logWindowSize } )
        else :
            print( 'WARNING> logger type %s not supported. Not logging training info' \
                   % self._loggerType )


    def train( self, args = {} ) :
        for iepisode in range( self._maxEpisodes ) :
            # reset the environment and grab initial observation
            _s = self._env.reset()
            # pass the agent some information about the start of an episode
            self._agent.onStartEpisode( { 's0' : _s } )
            # update this flag to end an episode
            _done = False

            while not _done :
                # grab an action from the agent
                _a = self._agent.act( _s )
                # and applied it into the world, and observe the results
                _snext, _r, _done, _ = self._env.step( _a )
                # pass the agent the step information for his own usage
                self._agent.update( ( _s, _a, _r, _snext, _done ) )
                # pass the logger the step information as well for his own usage
                if self._logger :
                    self._logger.update( ( _s, _a, _r, _snext, _done ) )

                # book keeping for next iteration
                _s = _snext

            # tell the agent the episode just finished
            self._agent.onEndEpisode()
            # and also the logger, to update its internal state
            self._logger.onEndEpisode()

    def test( self, args = {} ) :
        for iepisode in range( self._numTestEpisodes ) :
            # reset the environment and grab initial observation
            _s = self._env.reset()
            # update this flag to end an episode
            _done = False

            while not _done :
                # run inference on the agent
                _a = self._agent.act( _s )
                # and apply the action in the environment
                _snext, _r, _done, _ = self._env.step( _a )
                # render the environment
                self._env.render()
                # pass the logger the step information as well for his own usage
                if self._logger :
                    self._logger.update( ( _s, _a, _r, _snext, _done ) )

                # book keeping for next iteration
                _s = _snext

            # update the logger once the episode ended, and show results so far
            self._logger.onEndEpisode()
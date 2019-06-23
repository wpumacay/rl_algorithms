
import os
import sys
import abc
import numpy as np
from collections import deque


class Trainer( abc.ABC ) :
    r"""Base class for all trainers

    Trainers consist of object in charge of the training and testing 
    process. This provides a handy interface to handle your training
    loop. Some implementations of basic trainers are implemented below,
    but the user is advised to implement her|his own trainer according
    to its requirements. The main idea is to abstract the training|testing 
    loop boiler plate code away from the user such that she|he can focus
    only on the agent's implementation

    """
    def __init__( self, trainerConfig, envBuilder, agentBuilder ) :
        super( Trainer, self ).__init__()

        # save the configuration dictionary for future reference
        self._config = trainerConfig

        # grab some information from the trainer configuration
        self._maxEpisodes = self._config.get( 'maxEpisodes', 1000 )
        self._maxSteps = self._config.get( 'maxSteps', 1000 )
        self._logWindowSize = self._config.get( 'logWindowSize', 100 )
        self._seed = self._config.get( 'seed', 0 )


    @abc.abcstractmethod
    def init( self ) :
        r"""Abstract method for trainer initialization

        Derive and implement your own functionality for your trainer 
        initialization, according to your setup requirements

        """
        pass


    @abc.abcstractmethod
    def train( self, args = {} ) :
        r"""Abstract method to start the training process

        Derive and implement your own functionality for your trainer 
        training process, according to your setup requirements

        """
        pass


    @abc.abcstractmethod
    def test( self, args = {} ) :
        r"""Abstract method to start the testing process

        Derive and implement your own functionality for your trainer 
        testint process, according to your setup requirements

        """
        pass


class SimpleTrainer( Trainer ) :
    r"""Creates a simple single-agent single-environment non-parallel trainer
    
    This is the most basic trainer, derived from the Trainer interface. Use it
    for simple setups that require only one agent and one environment running
    in a single process (no parallelism).

    Args:
        trainerConfig (dict): configuration for the trainer
        envBuilder (function): factory method to create the environment
        agentBuilder (function): factory method to create the agent

    """
    def __init__( self, trainerConfig, envBuilder, agentBuilder ) :
        super( SimpleTrainer, self ).__init__( trainerConfig )

        self._envBuilder = envBuilder
        self._agentBuilder = agentBuilder

        # single environment to be used for our setup
        self._env = None
        # single agent to be used for our setup
        self._agent = None


    def init( self ) :
        self._env = self._envBuilder()
        self._agent = self._agentBuilder()

        assert self._env is not None, 'ERROR> Could not create the environment'
        assert self._agent is not None, 'ERROR> Could not create the agent'

        self._env.seed( self._seed )
        self._agent.seed( self._seed )


    def train( self, args ) :
        pass
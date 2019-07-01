
import os
import sys
import abc
import random
import numpy as np
from collections import deque

from tqdm import tqdm, trange
from rl.utils.config import TrainerConfig
from rl.utils.loggers import LoggerTqdm
from rl.utils.loggers import LoggerFile

from IPython.core.debugger import set_trace

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
        self._loggerFile        = self._config.loggerFile
        self._seed              = self._config.seed
        self._mode              = self._config.mode
        self._testOnceTrained   = self._config.testOnceTrained
        self._numTestEpisodes   = self._config.numTestEpisodes
        self._modelFilename     = self._config.modelFilename


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
        # seed numpy and random-module random generators
        np.random.seed()
        random.seed()

        # seed both the environment and the agent
        self._env.seed( self._seed )
        self._agent.seed( self._seed )

        if self._loggerType == 'tqdm' :
            self._logger = LoggerTqdm( { 'maxEpisodes' : self._maxEpisodes if self._mode == 'train' else self._numTestEpisodes,
                                         'logWindowSize' : self._logWindowSize } )
        else :
            print( 'WARNING> logger type %s not supported. Not logging training info' \
                   % self._loggerType )


    def train( self, args = {} ) :
        self._agent.setMode( 'train' )
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

        self._agent._model.save( self._modelFilename )


    def test( self, args = {} ) :
        self._agent._model.load( self._modelFilename )

        self._agent.setMode( 'test' )
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


class PopulationTrainer( SimpleTrainer ) :
    r"""A simple single-(population-agent) single-environment non-parallel trainer
    
    This trainer is similar to the SimpleTrainer, but it has some handy variations
    used for population based agents, as the training loop differs a little

    Args:
        trainerConfig (TrainerConfig): configuration for the trainer
        env (Environment): environment to run training on
        agent (Agent): agent to train

    """
    def __init__( self, trainerConfig, env, agent ) :
        super( PopulationTrainer, self ).__init__( trainerConfig, env, agent )

        # size of the population to be used
        self._populationSize = self._config.populationSize

        # sanity check, remind the user that we are expecting a population
        assert self._populationSize != -1, 'ERROR> using a population based trainer \
                                            with no population (-1)'


    def train( self, args = {} ) :
        for iepisode in range( self._maxEpisodes ) :
            ####################################################################
            # run first over the whole population, agent should handle the ...
            # indexing of which sample is being used from the population
            self._agent.setMode( 'train' )
            for ipop in trange( self._populationSize, desc = 'Population>' ) :
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

                    # book keeping for next iteration
                    _s = _snext

                # tell the agent the episode just finished
                self._agent.onEndEpisode()
            ####################################################################

            ####################################################################
            # run in evaluation mode to get progress info for the logger
            self._agent.setMode( 'test' )
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

            # and also the logger, to update its internal state
            self._logger.onEndEpisode()
            ####################################################################

        self._agent._model.save( self._modelFilename )


class ParallelPopulationTrainer( Trainer ) :
    r"""Creates a simple single-agent single-environment parallel trainer using MPI
    
    This is the most basic parallel trainer, derived from the Trainer interface. 
    Use it for setups using population based methods, which can be easily parallelized
    across a set of processes. Each trainer object is created and handled by a specific 
    process.

    Args:
        trainerConfig (TrainerConfig): configuration for the trainer
        env (Environment): environment to run training on
        agent (Agent): agent to train
        rank (int): id of the process associated with this trainer
        numWorkers (int): number of processes used for training (each running a trainer)
        workersSeeds (list): all seeds used for all other generators
        mpicomm (MPI.Intracom): communicator used by mpi

    """
    def __init__( self, trainerConfig, env, agent, rank, numWorkers, workersSeeds, mpicomm ) :
        super( ParallelPopulationTrainer, self ).__init__( trainerConfig )

        # single environment to be used for our setup
        self._env = env
        # single agent to be used for our setup
        self._agent = agent
        # rank|indx of the process associated with this trainer among all trainers
        self._rank = rank
        # number of workers used for training
        self._numWorkers = numWorkers
        # seeds associated with each worker
        self._workersSeeds = workersSeeds
        # grab the seed for this process
        self._seed = self._workersSeeds[self._rank]
        # store the mpi communicator we will use
        self._mpicomm = mpicomm

        assert self._env is not None, 'ERROR> No environment given for trainer'
        assert self._agent is not None, 'ERROR> No agent given for trainer'

        # size of the population to be used
        self._populationSize = self._config.populationSize

        # sanity check, remind the user that we are expecting a population
        assert self._populationSize != -1, 'ERROR> using a population based trainer \
                                            with no population (-1)'

        # chunk size of samples for each trainer in each process
        self._chunkPopulationSize = int( np.ceil( self._config.populationSize / numWorkers ) )

        # logger to be used for saving training information
        self._logger = None

        # progress bar for population
        self._progressBarPop = None


    def init( self ) :
        # seed numpy and random-module random generators
        np.random.seed()
        random.seed()

        # seed both the environment and the agent
        self._env.seed( self._seed )
        self._agent.seed( self._seed )

        # only make rank-0 process log the results
        if self._loggerType == 'tqdm' :
            if self._rank == 0 :
                self._logger = LoggerTqdm( { 'maxEpisodes' : self._maxEpisodes if self._mode == 'train' else self._numTestEpisodes,
                                             'logWindowSize' : self._logWindowSize } )
            else :
                self._logger = None
        elif self._loggerType == 'file' :
            if self._rank == 0 :
                self._logger = LoggerFile( { 'maxEpisodes' : self._maxEpisodes if self._mode == 'train' else self._numTestEpisodes,
                                             'logWindowSize' : self._logWindowSize,
                                             'filename' : self._loggerFile } )
            else :
                self._logger = None
        else :
            print( 'WARNING> logger type %s not supported. Not logging training info' \
                   % self._loggerType )


    def train( self, args = {} ) :

        for iepisode in range( self._maxEpisodes ) :
##             # @DEBUG: Check if all models over all processes have the same weights
##             _thisWeights = self._agent._meanModel.weights()
##     
##             _allOtherModelsWeights = self._mpicomm.allgather( _thisWeights )
##             if self._agent.checkForSameModels( _allOtherModelsWeights ) :
##                 _msg = 'INFO> Ok!, seems all models have the same initial weights\n\r'
##                 print( _msg )
##             else :
##                 _msg = 'INFO> NOOO!!!, it seems some models are different to this one\n\r'
##                 print( _msg )
##                 sys.exit( -1 )
##     
##             if self._rank == 0 :
##                 with open( 'checks.log', 'a' ) as fhandle :
##                     fhandle.write( _msg )

            ####################################################################
            # run first over the corresponding chunk size. The agent should ...
            # handle the indexing of which sample is being used from the ...
            # chunk-population-size
            
            _retInformation = {}
            self._agent.setMode( 'train' )

            self._progressBarPop = tqdm( range( self._chunkPopulationSize ), desc = 'Population>' ) \
                                        if self._rank == 0 else None

            for ipop in range( self._chunkPopulationSize ) :
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

                    # book keeping for next iteration
                    _s = _snext

                if self._progressBarPop :
                    self._progressBarPop.update()

                # tell the agent the episode just finished, and grab some ...
                # information if the agent has something to say
                _retInformation = self._agent.onEndEpisode()
            ####################################################################

            ####################################################################
            # gather all required information from all other processes

            assert _retInformation.get( 'chunkFinished', False ), \
                   'ERROR> the agent should have ended processing the chunk by now'

            assert ( 'chunkScores' in _retInformation ), \
                   'ERROR> should have scores returned by now'

            # grab scores obtained by this process over its own chunk
            _chunkScores = _retInformation['chunkScores']

            # gather all scores
            _allChunksScores = self._mpicomm.allgather( _chunkScores )
            _allScores = []
            for chunk in _allChunksScores :
                _allScores = _allScores + chunk

##             # @DEBUG
##             if self._rank == 0 :
##                 print( '*******************************************' )
##                 print( 'ChunkScores:' )
##                 print( _chunkScores )
##                 print( 'AllScores:' )
##                 print( _allScores )
##                 print( '*******************************************' )
## 
##             # @DEBUG
##             sys.exit( 0 )

            # run an update on the model once all scores have been gathered
            self._agent.onGathered( _allScores )

            ####################################################################

            if self._rank == 0 :
                ################################################################
                # run in evaluation mode (if rank-0) to get progress info for logger
                self._agent.setMode( 'test' )
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

                # and also the logger, to update its internal state
                if self._logger :
                    self._logger.onEndEpisode()
                ################################################################

        # only save model once, and make sure mean-model is copied to main-model
        if self._rank == 0 :
            self._agent._model.save( self._modelFilename )


    def test( self, args = {} ) :
        # only allow rank-0 process to test once finished
        if self._rank != 0 :
            return

        self._agent._model.load( self._modelFilename )

        self._agent.setMode( 'test' )
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
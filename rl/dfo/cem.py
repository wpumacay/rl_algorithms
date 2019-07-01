
import numpy as np
from rl.dfo.agent import DFOAgent

from IPython.core.debugger import set_trace

ERROR_MSG_KEY_ERROR = 'ERROR> key (%s) should have been provided'

class CEMAgent( DFOAgent ) :


    def __init__( self, name, config, model ) :
        super( CEMAgent, self ).__init__( name, config, model )

        # parameters for the gaussian perturbation
        self._noiseScale = config.eps0 * config.sigma0
        # score of a trajectory
        self._score = 0.
        # best score so far
        self._bestScore = -np.inf
        # discount factor
        self._gamma = config.gamma
        # step counter
        self._istep = 0

        # number of samples from the distribution to be evaluated
        self._populationSize = config.populationSize
        # number of samples to keep for refitting the distribution
        self._elitesSize = int( np.ceil( config.elitesFraction * config.populationSize ) )

        # an indicator of the current sample being evaluated
        self._currentSampleIndx = 0

        # model representing the mean of the distribution
        self._meanModel = self._model.clone()
        self._meanModel.seed( self._seed )

        # states of the random number generator of the model (self._model) when applying each perturbation
        self._randStates = []
        # all scores saved so far
        self._scores = []
    

    def onStartEpisode( self, args = {} ) :
        if self._mode == 'train' :
            # perturb the mean-model to generate a sample for evaluation
            self._model.copy( self._meanModel )
            _befState, _nowState = self._model.perturb( 'gaussian', 
                                                        { 'perturbationScale' : self._noiseScale } )

            # save the random state just before perturbation for later reconstruction
            self._randStates.append( _befState )
        else :
            # use the mean model for evaluation
            self._model.copy( self._meanModel )


    def update( self, transition ) :
        # unpack the reward from the transition ( s, a, r, s', done )
        _, _, _r, _, _ = transition
        # update the score, discounting it appropriately
        self._score += (self._gamma ** self._istep) * _r
        # and also update the step counter
        self._istep += 1


    def onEndEpisode( self, args = {} ) :
        if self._mode == 'train' :
            self._currentSampleIndx = ( self._currentSampleIndx + 1 ) % self._populationSize
            self._scores.append( self._score )
    
            if self._currentSampleIndx == 0 :
                # we have finished a pass, so update the mean-model from elites
                _elitesIndices = np.argsort( self._scores )[-self._elitesSize:]
                self._bestScore = max( self._scores )
                _elitesRandStates = [ self._randStates[i] for i in _elitesIndices ]
                # recompute the mean-model from the elites perturbed states
                for _rstate in _elitesRandStates :
                    self._meanModel.perturb( 'gaussian', 
                                             { 'perturbationScale' : ( self._noiseScale / self._elitesSize ),
                                               'randState' : _rstate } )
    
                # reset the scores|randStates buffer for another pass
                self._scores = []
                self._randStates = []

        # clear counters and accumulators for the next iteration
        self._istep = 0
        self._score = 0.


    @property
    def noiseScale( self ) :
        return self._noiseScale
    

class CEMAgentParallel( DFOAgent ) :


    def __init__( self, name, config, model, rank, numWorkers, seeds ) :
        super( CEMAgentParallel, self ).__init__( name, config, model )

        # id of the agent, related to the process in which is being run
        self._rank = rank
        # number of workers used for training
        self._numWorkers = numWorkers
        # seeds of all other processes that contain the other agents
        self._seeds = seeds

        ## Same parameters as in non-parallel case #############################
        # parameters for the gaussian perturbation
        self._noiseScale = config.eps0 * config.sigma0
        # score of a trajectory
        self._score = 0.
        # best score so far
        self._bestScore = -np.inf
        # discount factor
        self._gamma = config.gamma
        # step counter
        self._istep = 0
        ########################################################################

        # total population size among all workers
        self._totalPopulationSize = config.populationSize
        # population size chunk assigned to this agent
        self._chunkPopulationSize = int( np.ceil( config.populationSize / numWorkers ) )

        # total elites size among all workers
        self._totalElitesSize = int( np.ceil( config.elitesFraction * config.populationSize ) )
        # elites size chunk assigned to this agent
        self._chunkElitesSize = int( np.ceil( config.elitesFraction * config.populationSize / numWorkers ) )

        # indicator of the current sample being in the chunk evaluated
        self._currentSampleIndx = 0

        # model representing the mean of the distribution
        self._meanModel = self._model.clone()
        self._meanModel.seed( self._seed )

        # random number generators of all workers
        self._randGens = [ np.random.RandomState( self._seeds[i] ) for i in range( self._numWorkers ) ]
        # states of all random number generators of the models in across all workers
        self._randStates = []

        # all scores of this chunk
        self._chunkScores = []


    def onStartEpisode( self, args = {} ) :
        if self._mode == 'train' :
            # perturb the mean-model to generate a sample for evaluation, also ...
            # we are not caching the random states, as these will be computed ...
            # again for all workers during the gather operation
            self._model.copy( self._meanModel )
            self._model.perturb( 'gaussian', 
                                 { 'perturbationScale' : self._noiseScale } )
        else :
            # use the mean model for evaluation
            self._model.copy( self._meanModel )


    def update( self, transition ) :
        # unpack the reward from the transition ( s, a, r, s', done )
        _, _, _r, _, _ = transition
        # update the score, discounting it appropriately
        self._score += (self._gamma ** self._istep) * _r
        # and also update the step counter
        self._istep += 1


    def onEndEpisode( self, args = {} ) :
        _retInformation = { 'chunkFinished' : False }

        if self._mode == 'train' :
            self._currentSampleIndx = ( self._currentSampleIndx + 1 ) % self._chunkPopulationSize
            self._chunkScores.append( self._score )
    
            if self._currentSampleIndx == 0 :
                # we are ready to start the gather process, so notify the ...
                # trainer and send the required information for all-gather step
                _retInformation['chunkFinished'] = True
                _retInformation['chunkScores'] = self._chunkScores

        # clear counters and accumulators for the next iteration
        self._istep = 0
        self._score = 0.

        return _retInformation


    def onGathered( self, allScores ) :
        r"""Method called once all-gather operation finishes

        This method should be called by the trainer to send the required
        score information from all workers (including this worker) once the
        all-gather operation finishes

        Args:
            allScores (list): scores across all workers, obtained from their interactions in each process

        """
        # 0) clean the buffer of the random states and the chunk scores
        self._randStates = []
        self._chunkScores = []

        # 1) Reconstruct all random states from all perturbations made across ...
        #    all workers. We do this using each random generator associated ...
        #    which each worker by unrolling over the chunk population size.
        #    We reconstruct each perturbation and store the random state for ...
        #    later usage, but do not use the random number generator as it ...
        #    has to be kept in sync with the other generators (run exactly chunkSize times)

        for i in range( self._numWorkers ) :
            for j in range( self._chunkPopulationSize ) :
                # recreate the random perturbations, but avoid applying it
                # recall that the random number generator was called exactly ...
                # chunkPopulationSize times (one per each sample in the chunk)
                _befRandState, _nowRandState = self._model.perturb( 'gaussian',
                                                                    { 'perturbationScale' : 0.0,
                                                                      'externRandGen' : self._randGens[i],
                                                                      'applyPerturbation' : False } )

                # store the random state used to create the perturbation
                self._randStates.append( _befRandState )

        _sgen1 = self._randGens[self._rank].get_state()
        _sgen2 = self._model._randgen.get_state()

        assert ( _sgen1[0] == _sgen2[0] and
                 np.array_equal( _sgen1[1], _sgen2[1] ) and
                 _sgen1[2] == _sgen2[2] and
                 _sgen1[3] == _sgen2[3] ), 'ERROR> random generators should land in same spot'

        ## print( 'rank %s: len(randstates) = %d, totalPopulationSize = %d' % ( self._rank, len( self._randStates ), self._totalPopulationSize ) )

        assert len( self._randStates ) == self._totalPopulationSize, \
               'ERROR> mismatch in separation of chunks.'

        # 2) Compute elites using all scores and reconstruct the perturbations ...
        #    of the elites according to the random states recomputed previously.
        #    Scores have to be in the following form, to ensure alginemnt with ...
        #    random states :
        #       
        #  allScores:       [ SCORES_WORKER_1 - SCORES_WORKER_2 - ... - SCORES_WORKER_N ]
        #  allRandStates:   [ RANDOM_STATES_1 - RANDOM_STATES_2 - ... - RANDOM_STATES_N ]

        _elitesIndices = np.argsort( allScores )[-self._totalElitesSize:]
        self._bestScore = max( allScores )
        ## print( 'rank %s: len(elitesIndices) = %d, len(randstates) = %d' % ( self._rank, len( _elitesIndices ), len( self._randStates ) ) )
        ## print( 'rank %s: elitesIndices = ', _elitesIndices )
        _elitesRandStates = [ self._randStates[i] for i in _elitesIndices ]
        _elitesWorker = [ eliteIndex // self._chunkPopulationSize for eliteIndex in _elitesIndices ]

        # 3) Update the mean-model from the elites and using the perturbations ...
        #    applied at each step for each elite-sample. Use the same-seed random ...
        #    generator of the mean model with the appropriate random state from before

        for _rstate, _workerId in zip( _elitesRandStates, _elitesWorker ) :
            self._meanModel.seed( self._seeds[_workerId] )
            self._meanModel.perturb( 'gaussian', 
                                     { 'perturbationScale' : ( self._noiseScale / self._totalElitesSize ),
                                       'randState' : _rstate } )


    def checkForSameModels( self, otherModelsWeights ) :
        r"""Checks if this models has same weights compared to other processes models

        Args:
            otherModelsWeights (list): a list with the weights of each model (each process)

        """
        _allEqual = True
        for i, _otherModelWeights in enumerate( otherModelsWeights ) :
            if not self._meanModel.areWeightsEqual( _otherModelWeights ) :
                _allEqual = False
                print( 'WARNING> models %d and %d have different weights' \
                       % ( self._rank, i ) )
                

        return _allEqual

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

        # an indicator of the current sample being 
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


    def __init__( self, name, config, model, rank, seeds ) :
        super( CEMAgentParallel, self ).__init__( name, config, model )

        # id of the agent, related to the process in which is being run
        self._rank = rank
        # seeds of all other processes that contain the other agents
        self._seeds = seeds
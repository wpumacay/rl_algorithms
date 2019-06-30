
import gin

@gin.configurable
class TrainerConfig( object ) :

    def __init__( self,
                  envType = 'gym',
                  envName = 'CartPole-v0',
                  maxEpisodes = 1000,
                  maxStepsPerEpisode = 1000,
                  seed = 0,
                  logWindowSize = 100,
                  loggerType = 'tqdm',
                  loggerFile = 'logs',
                  dlbackend = 'keras',
                  mode = 'train',
                  testOnceTrained = True,
                  numTestEpisodes = 10,
                  modelFilename = 'model0',
                  populationSize = -1,
                  numWorkers = -1 ) :
        self.envType = envType                          # name of the library from which we will create our environment
        self.envName = envName                          # name of the environment (from the library given by type)
        self.maxEpisodes = maxEpisodes                  # max. number of episodes to train the agent(s)
        self.maxStepsPerEpisode = maxStepsPerEpisode    # max. number of steps that an episode will have
        self.seed = seed                                # random seed for all training
        self.logWindowSize = logWindowSize              # number of episodes to wait until refreshing logs
        self.loggerType = loggerType                    # type of logger to use for training
        self.loggerFile = loggerFile                    # file where to save logs in case of using a file-logger
        self.dlbackend = dlbackend                      # deep-learning backend to use
        self.mode = mode                                # whether to train or test
        self.testOnceTrained = testOnceTrained          # whether or not to test once training has finished
        self.numTestEpisodes = numTestEpisodes          # number of episodes to run during testing
        self.modelFilename = modelFilename              # filename used to save/load a trained model
        self.populationSize = populationSize            # size of the population used by the specific pop.based alg., -1 if not applicable
        self.numWorkers = numWorkers                    # number of workers used in case of parallel training is used, -1 if not applicable
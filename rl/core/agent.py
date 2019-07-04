
import abc


class Agent( abc.ABC ) :
    r"""Interface for all agents

    """
    def __init__( self, name, config ) :
        super( Agent, self ).__init__()

        self._name = name
        self._config = config
        self._seed = 0
        self._mode = 'train'


    @abc.abstractmethod
    def act( self, state ) :
        r"""Returns an a=action from the agent at state s=state
            
        Args:
            state (np.ndarray): state representation

        """
        pass


    @abc.abstractmethod
    def seed( self, seed = 0 ) :
        r"""Seeds the agent with a given seed

        Args:
            seed (int): seed for the agent's internal random machinery

        """
        pass


    @abc.abstractmethod
    def setMode( self, mode ) :
        r"""Sets the mode the agent is going to be used in (train|test)

        Args:
            mode (str): mode (train|test) which the agent will be set to

        """
        pass


    @abc.abstractmethod
    def update( self, transition ) :
        r"""Update agent's internals when a step is made in the environment

        Args:
            transition (tuple): a transition tuple ( s, a, r, s', done )

        """
        pass


    @abc.abstractmethod
    def clone( self ) :
        r"""Creates a deep clone of this agent
        

        """
        pass


    @property
    def name( self ) :
        return self._name


    @property
    def config( self ) :
        return self._config


    @property
    def seed( self ) :
        return self._seed


    @property
    def mode( self ) :
        return self._mode
    
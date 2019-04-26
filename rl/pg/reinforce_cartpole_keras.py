
import gym
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class Policy( object ) :

    def __init__( self, dimInput, dimOutput, dimHidden = 16 ) :
        super( Policy, self ).__init__()

        self.m_layers = {}
        self.m_layers['fc1'] = layers.Dense( dimHidden, activation = 'relu' )
        self.m_layers['fc2'] = layers.Dense( dimOutput, activation = 'softmax' )
        self.m_network = models.Sequential()
        self.m_network.add( self.m_layers['fc1'] )
        self.m_network.add( self.m_layers['fc2'] )

    def act( self, x ) :
        pass

    def fit( self ) :
        pass



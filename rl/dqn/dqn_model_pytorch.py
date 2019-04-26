
from rl.dqn.core.dqn_model_interface import IDqnModel

import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NetworkPytorchGeneric( nn.Module ) :

    def __init__( self, inputShape, outputShape, layersDefs ) :
        super( NetworkPytorchGeneric, self ).__init__()

        self._inputShape = inputShape
        self._outputShape = outputShape

        self._layersDefs = layersDefs.copy()
        self._layers = []

        self._build()

    def _build( self ) :
        _currShape = self._inputShape

        for layerDef in self._layersDefs :
            _layer, _currShape = self._createLayer( layerDef, _currShape )
            self._layers.append( _layer )

    def _createLayer( self, layerDef, currShape ) :
        _layer = None
        _nextShape = None
        
        if layerDef['type'] == 'fc' :
            # sanity check (should have a rank-1 tensor)
            assert len( currShape ) == 1, 'ERROR> must pass rank-1 tensor as input to fc layer'
            # grab the number of hidden units for this fc layer
            _nunits = layerDef['units']
            # and just create the layer
            _layer = nn.Linear( currShape[0], _nunits )
            _nextShape = ( _nunits )

        elif layerDef['type'] == 'conv2d' :
            # sanity check (should have at least a rank-2 tensor)
            assert len( currShape ) >= 2, 'ERROR> '
            

        elif layerDef['type'] == 'flatten' :
            _layer = lambda x : x.view( -1 )
            _nextShape = ( x.numel() )

        return _layer, _nextShape

    def forward( self, x ) :
        for stage in self._layers :
            # pass through current layer
            x = stage['layer'](x)

        return x

    def clone( self, other, tau ) :
        for _localParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _localParams.data.copy_( tau * _localParams.data + ( 1.0 - tau ) * _otherParams.data )

class NetworkTestLunarLander( nn.Module ) :

    def __init__( self, inputShape, outputShape ) :
        super( NetworkTestLunarLander, self ).__init__()

        # lunar lander has a 8-vector as an observation (rank-1 tensor)
        assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
        # and also has a discrete set of actions, with a 4-vector for its qvalues
        assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

        self._inputShape = inputShape
        self._outputShape = outputShape

        # define layers for this network
        self.fc1 = nn.Linear( self._inputShape[0], 128 )
        self.fc2 = nn.Linear( 128, 64 )
        self.fc3 = nn.Linear( 64, 16 )
        self.fc4 = nn.Linear( 16, self._outputShape[0] )

        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.out = None

    def forward( self, X ) :
        self.h1 = F.relu( self.fc1( X ) )
        self.h2 = F.relu( self.fc2( self.h1 ) )
        self.h3 = F.relu( self.fc3( self.h2 ) )

        self.out = self.fc4( self.h3 )

        return self.out

    def clone( self, other, tau ) :
        for _localParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _localParams.data.copy_( tau * _localParams.data + ( 1.0 - tau ) * _otherParams.data )

class DqnModelPytorch( IDqnModel ) :

    def __init__( self, modelConfig ) :
        super( DqnModelPytorch, self ).__init__( modelConfig )

        self._nnetwork = None
        self._optimizer = None
        self._lossFcn = nn.MSELoss()
        self._losses = deque( maxlen = 100 )

        self._device = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

    def build( self ) :
        # @TEST: creating a custom fc network for lunar lander
        self._nnetwork = NetworkTestLunarLander( self._inputShape,
                                                 self._outputShape,
                                                 self._layersDefs )

        # @TODO: Add optimizer options to modelconfig
        self._optimizer = optim.Adam( self._nnetwork.parameters(), lr = self._lr )

    def eval( self, state, inference = False ) :
        self._nnetwork.eval()

        return self._nnetwork.forward( state )

    def train( self, states, targets ) :
        self._nnetwork.train()
        
        _xx = torch.from_numpy( states ).to( self._device )
        _yy = torch.from_numpy( targets ).to( self._device )

        # reset the gradients buffer
        self._optimizer.zero_grad()

        # do forward pass to compute q-target predictions
        _yyhat = self._nnetwork.forward( _xx )

        # and compute loss and gradients
        _loss = self._lossFcn( _yyhat, _yy )
        _loss.backward()

        # run optimizer to update the weights
        self._optimizer.step()

        # grab loss for later statistics
        self._losses.append( _loss.item() )

    def clone( self, other, tau = 1.0 ) :
        self._nnetwork.clone( other, tau )

    def save( self, filename ) :
        pass

    def load( self, filename ) :
        pass

DqnModelBuilder = lambda config : DqnModelPytorch( config )
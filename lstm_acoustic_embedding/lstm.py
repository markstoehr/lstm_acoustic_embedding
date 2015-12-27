"""LSTM layer in theano."""

from __future__ import division
import numpy
import theano
from theano import tensor
from theano.tensor import nnet
import mlp
import utils
import pickle
import os
import copy

import data_io

import couscous.mlp as mlp
import couscous.theano_utils as theano_utils

SEED = 123
THEANOTYPE = theano.config.floatX
numpy.random.seed(SEED)

def batchify(xs):
    max_length = max(len(x) for x in xs)
    n_sequences = len(xs)
    n_dim = xs[0].shape[1]
    xs_array = numpy.empty((max_length, n_sequences, n_dim), dtype=THEANOTYPE)
    mask = numpy.zeros((max_length, n_sequences), dtype=THEANOTYPE)
    for x_index, x in enumerate(xs):
        x_length = len(x)
        xs_array[:x_length, x_index] = x
        mask[:x_length, x_index] = 1.0
    return xs_array, mask

class BatchMultiLayerLSTMNN(object):
    """
    LSTM with fully connected hidden layers on top.
    """
    def __init__(
            self, rng, input, mask, n_in, lstm_n_hiddens, mlp_hidden_specs, srng=None,
            lstm_parameters=None,
            mlp_parameters=None, output_type="last", prefix="lstms_mlp", truncate_gradient=-1):
        self.rng = rng
        self.srng = srng
        self.output_type = output_type
        self.input = input
        self.mask = mask
        self.n_in = n_in
        self.lstm_n_hiddens = lstm_n_hiddens
        self.mlp_hidden_specs = mlp_hidden_specs
        self.truncate_gradient = truncate_gradient
        self.layers = []
        self.l2 = 0.

        self.mlp_hidden_specs = mlp_hidden_specs
        for layer_spec in mlp_hidden_specs:
            mlp.activation_str_to_op(layer_spec)
            
        self.lstms = BatchMultiLayerLSTM(
            self.rng, self.input, self.mask, self.n_in, self.lstm_n_hiddens,
            parameters=lstm_parameters, output_type=self.output_type,
            prefix=prefix + "_lstms", truncate_gradient=self.truncate_gradient)

        self.l2 += self.lstms.l2
        self.parameters = self.lstms.parameters[:]
        self.lstm_parameters = self.lstms.parameters
        
        # get the mlp parameters set up so that we can determine initilization as needed
        if mlp_parameters is not None:
            mlp_parameters = mlp_parameters[:]
        else:
            mlp_parameters = None

        # these are loop constants that we update and keep track of
        cur_input = self.lstms.output
        cur_n_in = self.lstm_n_hiddens[-1]
        self.mlp_layers = []
        self.mlp_parameters = []
        for i_layer, layer_spec in enumerate(self.mlp_hidden_specs):
            if mlp_parameters is not None:
                W = mlp_parameters.pop(0)
                b = mlp_parameters.pop(0)
            else:
                W = None
                b = None

            try:
                layer =mlp.HiddenLayer(
                    rng=rng, input=cur_input, d_in=cur_n_in, d_out=layer_spec["units"],
                    activation=layer_spec["activation"], W=W, b=b)
            except: import pdb; pdb.set_trace()
            self.mlp_layers.append(layer)
            cur_input = layer.output
            cur_n_in = layer_spec["units"]
            self.mlp_parameters.extend([layer.W, layer.b])
            self.l2 += (layer.W**2).sum()
            
        self.output = cur_input
        self.layers.extend(self.lstms.layers[:])
        self.layers.extend(self.mlp_layers[:])
        self.parameters.extend(self.mlp_parameters[:])

    def save(self, f):
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        for layer in self.layers:
            layer.load(f)

class BatchMultiLayerLSTMMLP(object):
    def __init__(
            self, rng, input, mask, n_in, n_out, lstm_n_hiddens, mlp_hidden_specs, srng=None,
            lstm_parameters=None,
            mlp_parameters=None, output_type="last", prefix="lstms_mlp", truncate_gradient=-1):
        self.rng = rng
        self.srng = srng
        self.output_type = output_type
        self.input = input
        self.mask = mask
        self.n_in = n_in
        self.n_out = n_out
        self.lstm_n_hiddens = lstm_n_hiddens
        self.mlp_hidden_specs = mlp_hidden_specs
        self.truncate_gradient = truncate_gradient
        self.layers = []
        self.l2 = 0.
        self.lstms = BatchMultiLayerLSTM(
            self.rng, self.input, self.mask, self.n_in, self.lstm_n_hiddens,
            parameters=lstm_parameters, output_type=self.output_type,
            prefix=prefix + "_lstms", truncate_gradient=self.truncate_gradient)
        self.l2 += self.lstms.l2
        self.parameters = self.lstms.parameters[:]
        self.mlp = mlp.MLP(
            self.rng, self.lstms.output, self.lstm_n_hiddens[-1], self.n_out, self.mlp_hidden_specs, srng)
        self.l2 += self.mlp.l2
        self.layers.extend(self.lstms.layers[:])
        self.layers.extend(self.mlp.layers[:])
        self.output = self.mlp.layers[-1].output
        self.parameters.extend(self.mlp.parameters[:])
        self.y_pred = self.layers[-1].y_pred
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

    def save(self, f):
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        for layer in self.layers:
            layer.load(f)
                        

class BatchMultiLayerLSTM(object):
    """
    LSTM with multiple layers.
    """
    def __init__(self, rng, input, mask, n_in, n_hiddens, parameters=None,
                 output_type="last", prefix="lstms", truncate_gradient=-1,
                 srng=None, dropout=0.0):
        self.output_type = output_type
        self.dropout = dropout
        self.truncate_gradient = truncate_gradient
        self.n_layers = len(n_hiddens)
        self.layers = []
        self.input = input
        self.mask = mask
        self.n_in = n_in
        self.prefix = prefix
        # reverse and copy because we want to pop off the parameters
        if parameters is not None:
            cur_parameters = list(parameters)[::-1]
        else:
            cur_parameters = None
            
        self.parameters = []
        cur_in = n_in
        self.l2 = 0.
        for layer_id, n_hidden in enumerate(n_hiddens):
            cur_output_type = output_type if layer_id == self.n_layers-1 else "all"
            if cur_parameters is None:
                W = None
                U = None
                b = None
            else:
                W = cur_parameters.pop()
                U = cur_parameters.pop()
                b = cur_parameters.pop()

            if self.layers:
                input = self.layers[-1].output
                
            self.layers.append(
                BatchLSTM(rng, input, mask, cur_in, n_hidden, W=W, U=U, b=b,
                          output_type=cur_output_type,
                          prefix="%s_%d" % (self.prefix, layer_id),
                     truncate_gradient=self.truncate_gradient))
            self.parameters.append(self.layers[-1].W)
            self.parameters.append(self.layers[-1].U)
            self.parameters.append(self.layers[-1].b)
            self.l2 += self.layers[-1].l2
            cur_in = n_hidden
        self.output = self.layers[-1].output
        if srng is not None and dropout > 0.0:
            self.dropout_output = theano_utils.apply_dropout(
                srng, self.output, p=dropout)


    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        for layer in self.layers:
            layer.load(f)


def BatchMultiLayerLSTMFactory(rng, n_in, n_hiddens, parameters=None,
                               output_type="last", prefix="lstms", truncate_gradient=-1):
    return lambda input, mask: BatchMultiLayerLSTM(
        rng, input, mask, n_in, n_hiddens, parameters=parameters,
        output_type=output_type, prefix=prefix,
        truncate_gradient=truncate_gradient)

class BatchLSTM(object):
    """
    LSTM with batch processing. Assumption is

    input: (n_time_steps, n_sequences, n_dim)
    """
    def __init__(self, rng, input, mask, n_in, n_hidden, W=None, U=None, b=None,
                 output_type="last", prefix="lstm", truncate_gradient=-1):
        self.truncate_gradient = truncate_gradient
        self.output_type = output_type
        self.input = input
        self.mask = mask
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.prefix = prefix
        if W is None or U is None or b is None:
            WU_values = numpy.concatenate(
                [ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ], axis=1)
            W_values = WU_values[:self.n_in]
            U_values = WU_values[self.n_in:]
            W = theano.shared(value=W_values, name="%s_W" % prefix,
                              borrow=True)
            U = theano.shared(value=U_values, name="%s_U" % prefix,
                              borrow=True)
            b_values = numpy.zeros(4 * self.n_hidden, dtype=THEANOTYPE)
            b = theano.shared(value=b_values, name="%s_b" % prefix,
                              borrow=True)
        self.W = W
        self.U = U
        self.b = b
        self.parameters = [self.W, self.U, self.b]
        self.l2 = (self.W**2).sum() + (self.U**2).sum()
            
        self.input = input
        self.set_output()

    def set_output(self):
        hidden_features = batch_lstm_function(self.input, self.mask, self.n_hidden, self.W, self.U, self.b,
                                        prefix=self.prefix, truncate_gradient=self.truncate_gradient)
        if self.output_type == "last":
            self.output = hidden_features[-1]
        elif self.output_type == "mean":
            self.output = hidden_features.mean(axis=0)
        elif self.output_type == "max":
            self.output = hidden_features.max(axis=0)
        else:
            self.output = hidden_features

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        pickle.dump(self.W.get_value(borrow=True), f, -1)
        pickle.dump(self.U.get_value(borrow=True), f, -1)
        pickle.dump(self.b.get_value(borrow=True), f, -1)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.W.set_value(pickle.load(f), borrow=True)
        self.U.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)
        # self.n_in, self.n_hidden = self.W.get_value(borrow=True).shape
        # self.parameters = [self.W, self.U, self.b]
        # self.set_output()

def batch_lstm_function(state_below, mask, n_hidden, W, U, b, prefix="lstm", truncate_gradient=-1):
    """
    state_below: (n_timesteps, n_sequences, n_dim)
    """
    n_steps = state_below.shape[0]
    n_sequences = state_below.shape[1]
    
    
    def _slice(_x, n, dim):
        return _x[:, n*dim:(n+1) * dim]

    def _step(x_, m_, h_, c_):
        preact = tensor.dot(h_, U)
        preact += x_

        i = nnet.sigmoid(_slice(preact, 0, n_hidden))
        f = nnet.sigmoid(_slice(preact, 1, n_hidden))
        o = nnet.sigmoid(_slice(preact, 2, n_hidden))
        c = tensor.tanh(_slice(preact, 3, n_hidden))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    init_hidden = tensor.alloc(numpy_floatX(0.),
                               n_sequences, n_hidden)
    state_below = tensor.dot(state_below, W) + b
    rval, updates = theano.scan(_step,
                                sequences=[state_below, mask],
                                outputs_info=[init_hidden,
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_sequences,
                                                           n_hidden)],
                                name=_p(prefix, '_layers'),
                                truncate_gradient=truncate_gradient)
    return rval[0]




class MultiLayerLSTM(object):
    """
    LSTM with multiple layers.
    """
    def __init__(self, rng, input, n_in, n_hiddens, parameters=None,
                 output_type="last", prefix="lstms", truncate_gradient=-1,
                 srng=None, dropout=0.0):
        self.n_layers = len(n_hiddens)
        self.layers = []
        self.input = input
        self.n_in = n_in
        self.prefix = prefix
        self.dropout = dropout
        # reverse and copy because we want to pop off the parameters
        if parameters is not None:
            cur_parameters = list(parameters)[::-1]
        else:
            cur_parameters = None
            
        self.parameters = []
        cur_in = n_in
        self.l2 = 0.
        for layer_id, n_hidden in enumerate(n_hiddens):
            cur_output_type = output_type if layer_id == self.n_layers-1 else "all"
            if cur_parameters is None:
                W = None
                U = None
                b = None
            else:
                W = cur_parameters.pop()
                U = cur_parameters.pop()
                b = cur_parameters.pop()

            if self.layers:
                input = self.layers[-1].output
                
            self.layers.append(
                LSTM(rng, input, cur_in, n_hidden, W=W, U=U, b=b, output_type=cur_output_type,
                     prefix="%s_%d" % (self.prefix, layer_id),
                     truncate_gradient=truncate_gradient))
            self.parameters.append(self.layers[-1].W)
            self.parameters.append(self.layers[-1].U)
            self.parameters.append(self.layers[-1].b)
            self.l2 += self.layers[-1].l2
            cur_in = n_hidden

        
        self.output = self.layers[-1].output
        if srng is not None and dropout > 0.0:
            self.dropout_output = theano_utils.apply_dropout(
                srng, self.output, p=dropout)
        

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        for layer in self.layers:
            layer.load(f)


class MultiLayerLSTMNN(object):
    """
    LSTM with fully connected hidden layers on top.
    """
    def __init__(
            self, rng, input, n_in, lstm_n_hiddens, mlp_hidden_specs, srng=None,
            lstm_parameters=None,
            mlp_parameters=None, output_type="last", prefix="lstms_mlp", truncate_gradient=-1):
        self.rng = rng
        self.srng = srng
        self.output_type = output_type
        self.input = input
        self.n_in = n_in
        self.lstm_n_hiddens = lstm_n_hiddens
        self.mlp_hidden_specs = mlp_hidden_specs
        self.truncate_gradient = truncate_gradient
        self.layers = []
        self.l2 = 0.

        self.mlp_hidden_specs = mlp_hidden_specs
        for layer_spec in mlp_hidden_specs:
            mlp.activation_str_to_op(layer_spec)
            
        self.lstms = MultiLayerLSTM(
            self.rng, self.input, self.n_in, self.lstm_n_hiddens,
            parameters=lstm_parameters, output_type=self.output_type,
            prefix=prefix + "_lstms", truncate_gradient=self.truncate_gradient)
        self.lstm_parameters = self.lstms.parameters
        self.l2 += self.lstms.l2
        self.parameters = self.lstms.parameters[:]

        
        # get the mlp parameters set up so that we can determine initilization as needed
        if mlp_parameters is not None:
            mlp_parameters = mlp_parameters[:]
        else:
            mlp_parameters = None

        # these are loop constants that we update and keep track of
        cur_input = self.lstms.output
        cur_n_in = self.lstm_n_hiddens[-1]
        self.mlp_layers = []
        self.mlp_parameters = []
        for i_layer, layer_spec in enumerate(self.mlp_hidden_specs):
            if mlp_parameters is not None:
                W = mlp_parameters.pop(0)
                b = mlp_parameters.pop(0)
            else:
                W = None
                b = None

            layer =mlp.HiddenLayer(
                rng=rng, input=cur_input, d_in=cur_n_in, d_out=layer_spec["units"],
                activation=layer_spec["activation"], W=W, b=b)
            self.mlp_layers.append(layer)
            cur_input = layer.output
            cur_n_in = layer_spec["units"]
            self.mlp_parameters.extend([layer.W, layer.b])
            self.l2 += (layer.W**2).sum()
            
        self.output = cur_input
        self.layers.extend(self.lstms.layers[:])
        self.layers.extend(self.mlp_layers[:])
        self.parameters.extend(self.mlp_parameters[:])

    def save(self, f):
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        for layer in self.layers:
            layer.load(f)

            
class MultiLayerLSTMMLP(object):
    def __init__(
            self, rng, input, n_in, n_out, lstm_n_hiddens, mlp_hidden_specs, srng=None,
            lstm_parameters=None,
            mlp_parameters=None, output_type="last", prefix="lstms_mlp", truncate_gradient=-1):
        self.rng = rng
        self.srng = srng
        self.output_type = output_type
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.lstm_n_hiddens = lstm_n_hiddens
        self.mlp_hidden_specs = mlp_hidden_specs
        self.truncate_gradient = truncate_gradient
        self.layers = []
        self.lstms = MultiLayerLSTM(
            self.rng, self.input, self.n_in, self.lstm_n_hiddens,
            parameters=lstm_parameters, output_type=self.output_type,
            prefix=prefix + "_lstms", truncate_gradient=self.truncate_gradient)
        self.parameters = self.lstms.parameters[:]
        self.mlp = mlp.MLP(
            self.rng, self.lstms.output, self.lstm_n_hiddens[-1], self.n_out, self.mlp_hidden_specs, srng)
        self.layers.extend(self.lstms.layers[:])
        self.layers.extend(self.mlp.layers[:])
        self.output = self.mlp.layers[-1].output
        self.parameters.extend(self.mlp.parameters)
        self.y_pred = self.layers[-1].y_pred
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

    def save(self, f):
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        for layer in self.layers:
            layer.load(f)
                                    
class LSTM(object):
    """
    Long short term memory network.
    Attributes:
    """
    def __init__(self, rng, input, n_in, n_hidden, W=None, U=None, b=None,

                 output_type="last", prefix="lstm", truncate_gradient=-1):
        """
        initialization for hidden is just done at the zero level
        """
        self.truncate_gradient = truncate_gradient
        self.output_type = output_type
        self.input = input
        self.n_hidden = n_hidden
        self.n_in = n_in
        self.prefix = prefix
        if W is None or U is None or b is None:
            WU_values = numpy.concatenate(
                [ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + self.n_in)[:,
                                                               :self.n_hidden],
                 ], axis=1)
            W_values = WU_values[:self.n_in]
            U_values = WU_values[self.n_in:]
            W = theano.shared(value=W_values, name="%s_W" % prefix,
                              borrow=True)
            U = theano.shared(value=U_values, name="%s_U" % prefix,
                              borrow=True)
            b_values = numpy.zeros(4 * self.n_hidden, dtype=THEANOTYPE)
            b = theano.shared(value=b_values, name="%s_b" % prefix,
                              borrow=True)
        self.W = W
        self.U = U
        self.b = b
        self.parameters = [self.W, self.U, self.b]
        self.l2 = (self.W**2).sum() + (self.U**2).sum()
            
        self.input = input
        self.set_output()

    def set_output(self):
        hidden_features = lstm_function(self.input, self.n_hidden, self.W, self.U, self.b,
                                        prefix=self.prefix, truncate_gradient=self.truncate_gradient)
        if self.output_type == "last":
            self.output = hidden_features[-1]
        elif self.output_type == "max":
            self.output = hidden_features.max(axis=0)
        elif self.output_type == "mean":
            self.output = hidden_features.mean(axis=0)
        else:
            self.output = hidden_features

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        pickle.dump(self.W.get_value(borrow=True), f, -1)
        pickle.dump(self.U.get_value(borrow=True), f, -1)
        pickle.dump(self.b.get_value(borrow=True), f, -1)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.W.set_value(pickle.load(f), borrow=True)
        self.U.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)
        # self.n_in, self.n_hidden = self.W.get_value(borrow=True).shape
        # self.parameters = [self.W, self.U, self.b]
        # self.set_output()
        
            
        
def lstm_function(state_below, n_hidden, W, U, b, prefix="lstm", truncate_gradient=-1):
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1) * dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, U)
        preact += x_

        i = nnet.sigmoid(_slice(preact, 0, n_hidden))
        f = nnet.sigmoid(_slice(preact, 1, n_hidden))
        o = nnet.sigmoid(_slice(preact, 2, n_hidden))
        c = tensor.tanh(_slice(preact, 3, n_hidden))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)
        return h, c

    init_hidden = tensor.alloc(numpy_floatX(0.), n_hidden)
    state_below = tensor.dot(state_below, W) + b
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[init_hidden,
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_hidden)],
                                name=_p(prefix, '_layers'),
                                truncate_gradient=truncate_gradient)
    return rval[0]


        
def symbolic_multilayer_lstm(input, parameters, n_hiddens, hidden_acts,
                             init_hidden=None, prefix="lstm"):
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1) * dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(tensor.concatenate((h_, input_layer(x_, h_))), W)
        preact += b

        i = nnet.sigmoid(_slice(preact, 0, n_hidden))
        f = nnet.sigmoid(_slice(preact, 1, n_hidden))
        o = nnet.sigmoid(_slice(preact, 2, n_hidden))
        c = nnet.sigmoid(_slice(preact, 3, n_hidden))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)
        return h, c

    if init_hidden is None:
        init_hidden = tensor.alloc(numpy_floatX(0.), n_hidden)

    rval, updates = theano.scan(_step,
                                sequences=[input],
                                outputs_info=[init_hidden,
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_hidden)],
                                name=_p(prefix, '_layers'))
    return rval[0]

            
            
class LSTMLayer(object):
    """LSTM class."""

    def __init__(self, rng, input, input_frame_shape, n_hidden,
                 init_hidden=None,
                 input_layer=None, input_parameters=None, W=None,
                 b=None, prefix="lstm"):
        """Initialize symbolic parameters and expressions for LSTM.  Transition layer is
        a potentially non-linear combination of the current LSTM hidden state
        and the observation vector, this allows us to handle a 2D lstm and multiple iterations.

        transition_layer should be a function that takes input, trans_W, trans_b, activation=None, prefix
        """
        self.n_hidden = n_hidden
        self.prefix = prefix
        
        if input_layer is None:
            input_layer = lambda input_frame, hidden_frame: tensor.flatten(input_frame)
            input_parameters = []
            
        self.input_layer = input_layer
        self.input_parameters = input_parameters
        
        if W is None:
            input_frame_dim = int(numpy.prod(input_frame_shape))
            W_values = numpy.concatenate(
                [ortho_weight(self.n_hidden + input_frame_dim)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + input_frame_dim)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + input_frame_dim)[:,
                                                               :self.n_hidden],
                 ortho_weight(self.n_hidden + input_frame_dim)[:,
                                                               :self.n_hidden],
                 ], axis=1)
            
            W = theano.shared(value=W_values, name="%s_W" % prefix,
                              borrow=True)
            
        self.W = W

        if b is None:
            b_values = numpy.zeros(4 * self.n_hidden, dtype=THEANOTYPE)
            b = theano.shared(value=b_values, name="%s_b" % prefix,
                              borrow=True)
        self.b = b
        self.lstm_parameters = [self.W, self.b] + self.input_parameters
        self.input = input
        self.init_hidden = init_hidden
        self.output = symbolic_lstm(self.input, self.W, self.b, self.n_hidden,
                                    self.input_layer,
                                    init_hidden=self.init_hidden)


def symbolic_lstm(input, W, b, n_hidden, input_layer, init_hidden=None, prefix="lstm"):
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1) * dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(tensor.concatenate((h_, input_layer(x_, h_))), W)
        preact += b

        i = nnet.sigmoid(_slice(preact, 0, n_hidden))
        f = nnet.sigmoid(_slice(preact, 1, n_hidden))
        o = nnet.sigmoid(_slice(preact, 2, n_hidden))
        c = nnet.sigmoid(_slice(preact, 3, n_hidden))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)
        return h, c

    if init_hidden is None:
        init_hidden = tensor.alloc(numpy_floatX(0.), n_hidden)

    rval, updates = theano.scan(_step,
                                sequences=[input],
                                outputs_info=[init_hidden,
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_hidden)],
                                name=_p(prefix, '_layers'))
    return rval[0]


def numpy_floatX(data):
    return numpy.asarray(data, dtype=THEANOTYPE)

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params = param_init_lstm(options,
                             params,
                             prefix=options['lstm'])
    return params

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(THEANOTYPE)

def _p(pp, name):
    return '%s_%s' % (pp, name)


def param_init_lstm(options, params, prefix='lstm'):
    """Init the LSTM parameters."""
    W = numpy.concatenate([ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim']),
                           ortho_weight(options['n_hidden_dim'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['n_hidden_dim'],))
    params[_p(prefix, 'b')] = b.astype(THEANOTYPE)

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk, borrow=True)
    return tparams


def lstm_layer(tparams, state_below, init_state, options, prefix='lstm', mask=None):
    """Compute LSTM over a sequence and producing."""
    n_data = state_below.shape[0]
    
    def _slice(_x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['n_hidden_dim']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['n_hidden_dim']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['n_hidden_dim']))
        c = tensor.tanh(_slice(preact, 3, options['n_hidden_dim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    n_hidden_dim = options['n_hidden_dim']
    rval, updates = theano.scan(_step,
                                sequences=state_below,
                                outputs_info=[init_state,
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_hidden_dim)],
                                name=_p(prefix, '_layers'))
    
    return rval


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    x = tensor.matrix('x', dtype=THEANOTYPE)
    h = tensor.vector('h', dtype=THEANOTYPE)
    # mask = tensor.matrix('mask', dtype=config.floatX)
    # y = tensor.vector('y', dtype='')

    n_data = x.shape[0]

    lstm_hidden = lstm_layer(tparams, x, h, options, prefix=options['lstm'])

    return x, h, lstm_hidden

def old_main():
    x, h, lstm_out, params, tparams = construct_lstm_1d()
    import pdb; pdb.set_trace()

    x = tensor.matrix('x', dtype='float32')
    h = tensor.vector('h', dtype='float32')
    rng = numpy.random.RandomState(0)
    ndim = 3
    W = theano.shared(rng.randn(ndim, ndim).astype(numpy.float32),
                      name="W", borrow=True)
    b = theano.shared(rng.randn(ndim).astype(numpy.float32), name="b",
                      borrow=True)
    components, updates = theano.scan(fn=lambda x, h: nnet.relu(tensor.dot(W,h) + x + b),
                outputs_info=h,
                                      sequences=x)
    calculate_hiddens = theano.function(inputs=[x, h], outputs=[components])
    ntimes = 39
    x = rng.randn(ntimes, ndim).astype(numpy.float32)
    h = rng.randn(ndim).astype(numpy.float32)
    hs = calculate_hiddens(x, h)


def construct_lstm_1d():
    params = init_params(model_options)
    tparams = init_tparams(params)

    x, h, lstm_out = build_model(tparams, model_options)
    return x, h, lstm_out, params, tparams

def lstm_numpy(x, W, U, b):
    z = numpy.dot(x, W) + b
    n_hidden = b.shape[0]/4
    h = numpy.zeros((x.shape[0], n_hidden), dtype=x.dtype)
    prev_h = numpy.zeros(n_hidden, dtype=x.dtype)
    prev_c = numpy.zeros(n_hidden, dtype=x.dtype)
    
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1) * dim]
    for n in range(len(h)):
        preact = numpy.dot(prev_h, U) + z[n]
        i = utils.sigmoid(_slice(preact, 0, n_hidden))
        f = utils.sigmoid(_slice(preact, 1, n_hidden))
        o = utils.sigmoid(_slice(preact, 2, n_hidden))
        c = utils.tanh(_slice(preact, 3, n_hidden))

        c = f * prev_c + i * c
        h[n] = o * utils.tanh(c)
        prev_c = c
        prev_h = h[n]
    return h

def test_lstm():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hidden = 2
    lstm = LSTM(rng, x, n_in, n_hidden, output_type="full")
    f = theano.function(inputs=[x], outputs=lstm.output)

    n_data = 10
    x0 = rng.randn(n_data, n_in).astype(THEANOTYPE)
    h0 = f(x0)
    W = lstm.W.get_value()
    U = lstm.U.get_value()
    b = lstm.b.get_value()
    h0_numpy = lstm_numpy(x0, W, U, b)
    numpy.testing.assert_array_almost_equal(h0, h0_numpy)

def test_multilstm():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hiddens = [10, 10]
    multi_lstm = MultiLayerLSTM(rng, x, n_in, n_hiddens, output_type="last")
    f = theano.function(inputs=[x], outputs=multi_lstm.output)

    n_data = 20
    x0 = rng.randn(n_data, n_in).astype(THEANOTYPE)
    h1 = f(x0)
    W0 = multi_lstm.layers[0].W.get_value()
    U0 = multi_lstm.layers[0].U.get_value()
    b0 = multi_lstm.layers[0].b.get_value()
    W1 = multi_lstm.layers[1].W.get_value()
    U1 = multi_lstm.layers[1].U.get_value()
    b1 = multi_lstm.layers[1].b.get_value()
    h0_numpy = lstm_numpy(x0, W0, U0, b0)
    h1_numpy = lstm_numpy(h0_numpy, W1, U1, b1)[-1]
    numpy.testing.assert_array_almost_equal(h1, h1_numpy)

    # testing the loss
    cost = tensor.sum(multi_lstm.output**2)
    loss = theano.function(inputs=[x], outputs=cost)
    gradient = tensor.grad(cost, multi_lstm.parameters)

def test_maxmultilstm():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hiddens = [10, 10]
    multi_lstm = MultiLayerLSTM(rng, x, n_in, n_hiddens, output_type="max")
    f = theano.function(inputs=[x], outputs=multi_lstm.output)

    n_data = 20
    x0 = rng.randn(n_data, n_in).astype(THEANOTYPE)
    h1 = f(x0)
    W0 = multi_lstm.layers[0].W.get_value()
    U0 = multi_lstm.layers[0].U.get_value()
    b0 = multi_lstm.layers[0].b.get_value()
    W1 = multi_lstm.layers[1].W.get_value()
    U1 = multi_lstm.layers[1].U.get_value()
    b1 = multi_lstm.layers[1].b.get_value()
    h0_numpy = lstm_numpy(x0, W0, U0, b0)
    h1_numpy = lstm_numpy(h0_numpy, W1, U1, b1).max(axis=0)
    numpy.testing.assert_array_almost_equal(h1, h1_numpy)

    # testing the loss
    cost = tensor.sum(multi_lstm.output**2)
    loss = theano.function(inputs=[x], outputs=cost)
    gradient = tensor.grad(cost, multi_lstm.parameters)


def test_multibatchlstm():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hiddens = [10, 10]
    multi_lstm = MultiLayerLSTM(rng, x, n_in, n_hiddens, output_type="last")
    f = theano.function(inputs=[x], outputs=multi_lstm.output)

    xs = tensor.tensor3("xs", dtype=THEANOTYPE)
    mask = tensor.matrix("mask", dtype=THEANOTYPE)

    multi_batchlstm = BatchMultiLayerLSTM(rng, xs, mask, n_in, n_hiddens, parameters=multi_lstm.parameters, output_type="last")
    fs = theano.function(inputs=[xs, mask], outputs=multi_batchlstm.output)
    
    
    sequence_lengths = [5, 10, 15, 20]
    xs0 = [rng.randn(n_data, n_in).astype(THEANOTYPE)
           for n_data in sequence_lengths]
    hs0 = numpy.asarray([f(x0) for x0 in xs0])
    xs_arr0, mask = batchify(xs0)
    hs1 = fs(xs_arr0, mask)
    numpy.testing.assert_array_almost_equal(hs0, hs1)

def test_multisaveload():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hiddens = [10, 10]
    multi_lstm = MultiLayerLSTM(rng, x, n_in, n_hiddens, output_type="last")
    f0 = theano.function(inputs=[x], outputs=multi_lstm.output)
    save_file = data_io.smart_open("model.pkl.gz", "wb")
    multi_lstm.save(save_file)
    save_file.close()

    n_in1 = 4
    n_hiddens1 = [11, 11]
    multi_lstm1 = MultiLayerLSTM(rng, x, n_in, n_hiddens, output_type="last")
    load_file = data_io.smart_open("model.pkl.gz", "rb")
    multi_lstm1.load(load_file)
    load_file.close()
    f1 = theano.function(inputs=[x], outputs=multi_lstm1.layers[0].output)
    
    n_data = 10
    x0 = rng.randn(n_data, n_in).astype(THEANOTYPE)
    import pdb; pdb.set_trace()

def test_saveload():
    rng = numpy.random.RandomState(0)
    x = tensor.matrix("x", dtype=THEANOTYPE)
    n_in = 3
    n_hidden = 10
    lstm = LSTM(rng, x, n_in, n_hidden, output_type="last")
    n_data = 10
    x0 = rng.randn(n_data, n_in).astype(THEANOTYPE)

    f0 = theano.function(inputs=[x], outputs=lstm.output)
    h0 = f0(x0)
    save_file = data_io.smart_open("model.pkl.gz", "wb")
    lstm.save(save_file)
    save_file.close()

    x1 = tensor.matrix("x1", dtype=THEANOTYPE)
    lstm1 = LSTM(rng, x1, n_in, n_hidden, output_type="last")
    load_file = data_io.smart_open("model.pkl.gz", "rb")
    f1 = theano.function(inputs=[x1], outputs=lstm1.output)
    h1 = f1(x0)
    lstm1.load(load_file)
    load_file.close()
    h2 = f1(x0)
    numpy.testing.assert_array_almost_equal(h0, h2)
    
def main():
    test_lstm()
    test_multilstm()
    test_maxmultilstm()
    test_saveload()
    test_multibatchlstm()
    
if __name__ == "__main__":
    main()

"""MLR"""

import cPickle
import numpy
import theano
from theano import tensor
from theano.tensor import nnet
import ops
import timeit

NNTYPE = "float32"

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=nnet.relu, prefix="hidden", batchnorm=False):
        """Generic hidden layer."""
        self.input = input
            
        if W is None:
            if activation == tensor.tanh:
                W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                    ), dtype=theano.config.floatX)
            elif activation == nnet.sigmoid:
                W_values = 4 * numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                    ), dtype=theano.config.floatX)
            elif activation == nnet.relu or activation == nnet.softmax or activation is None:
                W_values = numpy.asarray(0.01*rng.randn(n_in, n_out), dtype=theano.config.floatX)
            else:
                raise ValueError("Invalid activation: " + str(activation))
            W = theano.shared(value=W_values, name="%s_W" % prefix,
                              borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            # if activation == theano_utils.relu:
            #     b_values = 0.01*np.ones((d_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="%s_b" % prefix,
                              borrow=True)

        self.W = W
        self.b = b
        linear_output = tensor.dot(input, self.W) + self.b
        if batchnorm:
            lienar_output = ops.batchnorm(linear_output)

        self.output = (linear_output
                       if activation is None else activation(linear_output))
        self.parameters = [self.W, self.b]

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        pickle.dump(self.W.get_value(borrow=True), f, -1)
        pickle.dump(self.b.get_value(borrow=True), f, -1)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.W.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)



def makeHiddenLayer(input, layer_parameters, activation, prefix="hidden"):
    layer = HiddenLayer(None, input, None, None, W=layer_parameters[0],
                        b=layer_parameters[1], activation=activation,
                        prefix=prefix)
    return layer.output


class RegressionNetwork(object):
    def __init__(self, rng, input, n_in, n_hidden_list=[20, 20, 20], activations=[nnet.relu, nnet.relu, None], parameters=None, prefix="hnetwork", batchnorm=False):
        """Generic Deep Neural Network.
   
        The last number in n_hidden_list is the number of output units.  The
        last layer will often be a linear layer with no activation.

        """
        self.input = input
        self.n_in = n_in
        self.prefix = prefix
        self.output = input
        if parameters is None:
            assert len(n_hidden_list) == len(activations), "Should have the same number of hidden variables and activations"
            self.n_hidden_list = n_hidden_list
            self.activations = activations
            self.n_layers = len(self.n_hidden_list)
            self.layer_list = []
            layer_n_in = n_in
            parameters = []
            for layer_id, (layer_n_out, layer_activation) in enumerate(
                    zip(self.n_hidden_list, self.activations)):
                self.layer_list.append(
                    HiddenLayer(rng, self.output, layer_n_in, layer_n_out,
                                activation=layer_activation,
                                prefix="%s_hidden%d" % (self.prefix, layer_id), batchnorm=batchnorm))
                layer_n_in = layer_n_out
                self.output = self.layer_list[-1].output
                parameters.extend(self.layer_list[-1].parameters)
        else:
            assert len(parameters) == 2 * len(activations), "Should have twice as many parameter entries as activations"
            self.activations = activations
            self.n_layers = len(activations)
            self.layer_list = []
            self.n_hidden_list = []
            for layer_id, (layer_W, layer_b) in enumerate(zip(parameters[::2],
                                                    parameters[1::2])):
                layer_n_in, layer_n_out = layer_W.get_value().shape
                layer_activation = self.activations[layer_id]
                self.layer_list.append(
                    HiddenLayer(rng, self.output, layer_n_in, layer_n_out,
                                W=layer_W, b=layer_b,
                                activation=layer_activation,
                                prefix="%s_hidden%d" % (self.prefix, layer_id), batchnorm=batchnorm))
                self.output = self.layer_list[-1].output
                self.n_hidden_list.append(layer_n_out)
        self.parameters = parameters
        self.output = self.output



def test_one_layer():
    """Ensure that hidden layer and the hidden network are working."""
    rng = numpy.random.RandomState(0)
    x = tensor.vector("x", dtype=NNTYPE)
    n_in = 3
    n_out = 4
    hlayer = HiddenLayer(rng, x, n_in, n_out)
    f = theano.function(inputs=[x], outputs=hlayer.output)

    W = hlayer.W.get_value()
    b = hlayer.b.get_value()
    f_numpy = lambda x: numpy.maximum(0, numpy.dot(x, W) + b)
    x0 = rng.randn(n_in).astype(NNTYPE)
    numpy.testing.assert_array_almost_equal(f(x0), f_numpy(x0))

    xs = tensor.matrix("xs", dtype=NNTYPE)
    n_in = 3
    n_out = 4
    hlayer = HiddenLayer(rng, xs, n_in, n_out, W=hlayer.W.get_value(),
                         b=hlayer.b.get_value())
    fs = theano.function(inputs=[xs], outputs=hlayer.output)
    xs0 = rng.randn(2, n_in).astype(NNTYPE)

    numpy.testing.assert_array_almost_equal(f(xs0[0]), fs(xs0)[0])
    numpy.testing.assert_array_almost_equal(f(xs0[1]), fs(xs0)[1])

def test_multiple_layers():
    """Ensure the multilayer system is also working."""
    rng = numpy.random.RandomState(0)
    x = tensor.vector("x", dtype=NNTYPE)
    n_in = 3
    n_out = 4
    regressnetwork = RegressionNetwork(rng, x, n_in, n_hidden_list=[20,20,n_out],
                                       activations=[nnet.relu, nnet.relu, None],
                                       prefix="hnetwork")
    f = theano.function(inputs=[x], outputs=regressnetwork.output)
    f_numpy_list = []
    relu = lambda x: numpy.maximum(0, x)
    f_numpy0 = lambda x: relu(numpy.dot(x, regressnetwork.parameters[0].get_value()) + regressnetwork.parameters[1].get_value())
    f_numpy1 = lambda x: relu(numpy.dot(f_numpy0(x), regressnetwork.parameters[2].get_value()) + regressnetwork.parameters[3].get_value())
    f_numpy2 = lambda x: numpy.dot(f_numpy1(x), regressnetwork.parameters[4].get_value()) + regressnetwork.parameters[5].get_value()
    x0 = rng.randn(n_in).astype(NNTYPE)
    numpy.testing.assert_array_almost_equal(f(x0), f_numpy2(x0))

    xs = tensor.matrix("xs", dtype=NNTYPE)
    regressnetwork2 = RegressionNetwork(rng, xs, n_in, n_hidden_list=[20,20,n_out],
                                       activations=[nnet.relu, nnet.relu, None],
                                       parameters=regressnetwork.parameters,
                                       prefix="hnetwork")
    fs = theano.function(inputs=[xs], outputs=regressnetwork2.layer_list[0].output)
    f = theano.function(inputs=[x], outputs=regressnetwork.layer_list[0].output)
    xs0 = rng.randn(2, n_in).astype(NNTYPE)

    numpy.testing.assert_array_almost_equal(f(xs0[0]), fs(xs0)[0])
    numpy.testing.assert_array_almost_equal(f(xs0[1]), fs(xs0)[1])

    
def main():
    test_one_layer()
    test_multiple_layers()

    
if __name__ == "__main__":
    main()

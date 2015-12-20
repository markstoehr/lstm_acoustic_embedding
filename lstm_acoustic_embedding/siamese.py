"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy
import scipy.spatial.distance as distance
import theano
import theano.tensor as T
from theano import tensor
import lstm


# #-----------------------------------------------------------------------------#
# #                              SIAMESE CNN CLASS                              #
# #-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
#                          SIAMESE TRIPLET LSTM CLASS                          #
#-----------------------------------------------------------------------------#

THEANOTYPE = theano.config.floatX

class SiameseTripletBatchLSTM(object):
    """
    Siamese triplet convolutional neural network, allowing for hinge losses.

    An example use of this network is to train a metric on triplets A, B, X.
    Assume A-X is a same pair and B-X a different pair. Then a cost can be
    defined such that dist(A, X) > dist(B, X) by some margin. By convention the
    same-pair is taken as `x1_layers` and `x2_layers`, while the different pair
    is taken as `x1_layers` and `x3_layers`.

    Attributes
    ----------
    x1_layers : list of ConvMaxPoolLayer and HiddenLayer
        Attributes are similar to `SiameseCNN`, except that now there are
        three tied networks, and so there is `x1_layers`, `x2_layers` and
        `x3_layers`, with corresponding additional layers when using dropout.
    """

    def __init__(self, rng, input_x1, input_x2, input_x3, input_m1, input_m2, input_m3, n_in, n_hiddens):
        """
        Initialize symbolic parameters and expressions.

        Many of the parameters are identical to that of `cnn.build_cnn_layers`.
        Some of the other parameters are described below.

        Parameters
        ----------
        input_x1 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the first side of the Siamese network.
        input_x2 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the second side of the Siamese network, forming a
            same-pair with `input_x1`.
        input_x3 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the third side of the Siamese network, forming a
            different-pair with `input_x1`.
        """

        # Build common layers to which the Siamese layers are tied
        input = T.tensor3("x", dtype=THEANOTYPE)
        mask = T.matrix("m", dtype=THEANOTYPE)
        self.input = input
        self.mask = mask
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_layers = len(self.n_hiddens)
        self.lstms = lstm.BatchMultiLayerLSTM(
            rng, input, mask, n_in, n_hiddens, output_type="last", prefix="lstms")

        self.x1_lstms = lstm.BatchMultiLayerLSTM(
            rng, input_x1, input_m1, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x1")
        self.x2_lstms = lstm.BatchMultiLayerLSTM(
            rng, input_x2, input_m2, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x2")
        self.x3_lstms = lstm.BatchMultiLayerLSTM(
            rng, input_x3, input_m3, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x3")

        self.parameters = self.lstms.parameters
        self.l2 = self.lstms.l2
        self.output = self.lstms.output

    def loss_hinge_cos(self, margin=0.5):
        return _loss_hinge_cos(
            self.x1_lstms.output,
            self.x2_lstms.output,
            self.x3_lstms.output,
            margin
            )

    def cos_same(self):
        """
        Return symbolic expression for the mean cosine distance of the same
        pairs alone.
        """
        return T.mean(cos_distance(self.x1_lstms.output, self.x2_lstms.output))

    def cos_diff(self):
        """
        Return symbolic expression for the mean cosine distance of the
        different pairs alone.
        """
        return T.mean(cos_distance(self.x1_lstms.output, self.x3_lstms.output))

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        self.lstms.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.lstms.load(f)


class SiameseTripletLSTM(object):
    """
    Siamese triplet convolutional neural network, allowing for hinge losses.

    An example use of this network is to train a metric on triplets A, B, X.
    Assume A-X is a same pair and B-X a different pair. Then a cost can be
    defined such that dist(A, X) > dist(B, X) by some margin. By convention the
    same-pair is taken as `x1_layers` and `x2_layers`, while the different pair
    is taken as `x1_layers` and `x3_layers`.

    Attributes
    ----------
    x1_layers : list of ConvMaxPoolLayer and HiddenLayer
        Attributes are similar to `SiameseCNN`, except that now there are
        three tied networks, and so there is `x1_layers`, `x2_layers` and
        `x3_layers`, with corresponding additional layers when using dropout.
    """

    def __init__(self, rng, input_x1, input_x2, input_x3, n_in, n_hiddens):
        """
        Initialize symbolic parameters and expressions.

        Many of the parameters are identical to that of `cnn.build_cnn_layers`.
        Some of the other parameters are described below.

        Parameters
        ----------
        input_x1 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the first side of the Siamese network.
        input_x2 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the second side of the Siamese network, forming a
            same-pair with `input_x1`.
        input_x3 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the third side of the Siamese network, forming a
            different-pair with `input_x1`.
        """

        # Build common layers to which the Siamese layers are tied
        input = T.matrix("x", dtype=THEANOTYPE)
        self.input = input
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_layers = len(self.n_hiddens)
        self.lstms = lstm.MultiLayerLSTM(
            rng, input, n_in, n_hiddens, output_type="last", prefix="lstms")

        self.x1_lstms = lstm.MultiLayerLSTM(
            rng, input_x1, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x1")
        self.x2_lstms = lstm.MultiLayerLSTM(
            rng, input_x2, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x2")
        self.x3_lstms = lstm.MultiLayerLSTM(
            rng, input_x3, n_in, n_hiddens, parameters=self.lstms.parameters,
            output_type="last", prefix="lstms_x3")

        self.parameters = self.lstms.parameters
        self.output = self.lstms.output

    def loss_hinge_cos(self, margin=0.5):
        return _loss_hinge_cos(
            self.x1_lstms.output,
            self.x2_lstms.output,
            self.x3_lstms.output,
            margin
            )

    def cos_same(self):
        """
        Return symbolic expression for the mean cosine distance of the same
        pairs alone.
        """
        return T.mean(cos_distance(self.x1_lstms.output, self.x2_lstms.output))

    def cos_diff(self):
        """
        Return symbolic expression for the mean cosine distance of the
        different pairs alone.
        """
        return T.mean(cos_distance(self.x1_lstms.output, self.x3_lstms.output))

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        self.lstms.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.lstms.load(f)


#-----------------------------------------------------------------------------#
#                            LOSS UTILITY FUNCTIONS                           #
#-----------------------------------------------------------------------------#

def cos_similarity(x1, x2):
    return (
        T.sum(x1 * x2, axis=-1) /
        (x1.norm(2, axis=-1) * x2.norm(2, axis=-1))
        )
cos_distance = lambda x1, x2: (1. - cos_similarity(x1, x2)) / 2.


def _loss_cos_cos2(x1, x2, y):
    cos_cos2 = T.switch(
        y, (1. - cos_similarity(x1, x2)) / 2., cos_similarity(x1, x2)**2
        )
    return T.mean(cos_cos2)


def _loss_cos_cos(x1, x2, y):
    cos_cos = T.switch(
        y, (1. - cos_similarity(x1, x2)) / 2., (cos_similarity(x1, x2) + 1.0) / 2.
        )
    return T.mean(cos_cos)


def _loss_cos_cos_margin(x1, x2, y, margin):
    loss_same = (1. - cos_similarity(x1, x2)) / 2.
    loss_diff = T.maximum(0., (cos_similarity(x1, x2) + 1.0) / 2. - margin)
    cos_cos = T.switch(
        y, loss_same, loss_diff
        )
    return T.mean(cos_cos)


def _loss_euclidean_margin(x1, x2, y, margin):
    loss_same = ((x1 - x2).norm(2, axis=-1))**2
    loss_diff = (T.maximum(0., margin - (x1 - x2).norm(2, axis=-1)))**2
    return T.mean(
        T.switch(y, loss_same, loss_diff)
        )


def _loss_hinge_cos(x1, x2, x3, margin):
    return T.mean(T.maximum(
        0.,
        margin + cos_distance(x1, x2) - cos_distance(x1, x3)
        ))


#-----------------------------------------------------------------------------#
#                                TEST FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def np_loss_cos_cos2(x1, x2, y):
    assert x1.shape[0] == x2.shape[0] == y.shape[0]
    losses = []
    for i in xrange(x1.shape[0]):
        if y[i] == 1:
            # Data points are the same, use cosine distance
            loss = distance.cosine(x1[i], x2[i]) / 2.
            losses.append(loss)
        elif y[i] == 0:
            # Data points are different, use cosine similarity squared
            loss = (distance.cosine(x1[i], x2[i]) - 1)**2
            losses.append(loss)
        else:
            assert False
    return numpy.mean(losses)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def old_main():
    from cnn import np_cnn_layers_output
    from theano.tensor.shared_randomstreams import RandomStreams
    import itertools
    import numpy.testing as npt
    import theano_utils


    # Test `SiameseCNN`

    # Random number generators
    rng = numpy.random.RandomState(42)
    srng = RandomStreams(seed=42)

    # Generate random data
    n_data = 4
    n_pairs = 6
    height = 39
    width = 200
    in_channels = 1
    X = rng.randn(n_data, in_channels, height, width)
    Y = numpy.asarray(rng.randint(2, size=n_pairs), dtype=numpy.int32)
    print "Same/diff:", Y

    # Generate random pairs
    possible_pairs = list(itertools.combinations(range(n_data), 2))
    x1_indices = []
    x2_indices = []
    for i_pair in rng.choice(numpy.arange(len(possible_pairs)), size=n_pairs, replace=False):
        x1, x2 = possible_pairs[i_pair]
        x1_indices.append(x1)
        x2_indices.append(x2)
    x1_indices = numpy.array(x1_indices)
    x2_indices = numpy.array(x2_indices)
    print "x1 index: ", x1_indices
    print "x2 index: ", x2_indices

    # Setup Theano model
    batch_size = n_pairs
    input_shape = (batch_size, in_channels, height, width)
    conv_layer_specs = [
        {"filter_shape": (32, in_channels, 39, 9), "pool_shape": (1, 3)},
        ]
    hidden_layer_specs = [{"units": 128}]
    dropout_rates = None
    y = T.ivector("y")
    input_x1 = T.matrix("x1")
    input_x2 = T.matrix("x2")
    activation = theano_utils.relu
    model = SiameseCNN(
            rng, input_x1, input_x2, input_shape,
            conv_layer_specs, hidden_layer_specs,
            srng, dropout_rates=dropout_rates,
            activation=activation
        )
    loss = model.loss_cos_cos2(y)

    # Compile Theano function
    theano_siamese_loss = theano.function(
        inputs=[], outputs=loss,
        givens={
            input_x1: X.reshape((n_data, -1))[x1_indices],
            input_x2: X.reshape((n_data, -1))[x2_indices],
            y: Y
            },
        )
    theano_loss = theano_siamese_loss()
    print "Theano loss:", theano_loss

    # Calculate Numpy output
    conv_layers_W = []
    conv_layers_b = []
    conv_layers_pool_shape = []
    hidden_layers_W = []
    hidden_layers_b = []
    for i_layer in xrange(len(conv_layer_specs)):
        W = model.layers[i_layer].W.get_value(borrow=True)
        b = model.layers[i_layer].b.get_value(borrow=True)
        pool_shape = conv_layer_specs[i_layer]["pool_shape"]
        conv_layers_W.append(W)
        conv_layers_b.append(b)
        conv_layers_pool_shape.append(pool_shape)
    for i_layer in xrange(i_layer + 1, i_layer + 1 + len(hidden_layer_specs)):
        W = model.layers[i_layer].W.get_value(borrow=True)
        b = model.layers[i_layer].b.get_value(borrow=True)
        hidden_layers_W.append(W)
        hidden_layers_b.append(b)
    np_x1_layers_output = np_cnn_layers_output(
        X[x1_indices], conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b, activation=theano_utils.np_relu
        )
    np_x2_layers_output = np_cnn_layers_output(
        X[x2_indices], conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b, activation=theano_utils.np_relu
        )

    numpy_loss = np_loss_cos_cos2(np_x1_layers_output, np_x2_layers_output, Y)
    print "Numpy loss:", numpy_loss

    npt.assert_almost_equal(numpy_loss, theano_loss)


def main():
    x1 = tensor.tensor3("x1", dtype=THEANOTYPE)
    x2 = tensor.tensor3("x2", dtype=THEANOTYPE)
    x3 = tensor.tensor3("x3", dtype=THEANOTYPE)
    x1_indices = tensor.ivector("x1_indices")
    x2_indices = tensor.ivector("x2_indices")
    x3_indices = tensor.ivector("x3_indices")
    m1 = tensor.matrix("m1", dtype=THEANOTYPE)
    m2 = tensor.matrix("m2", dtype=THEANOTYPE)
    m3 = tensor.matrix("m3", dtype=THEANOTYPE)
    rng = numpy.random.RandomState(0)
    n_data = 1000
    max_sequence_length = 50
    n_dim = 5
    n_hiddens = [10, 10]
    model = SiameseTripletBatchLSTM(
        rng, x1, x2, x3, m1, m2, m3, n_in=n_dim, n_hiddens=n_hiddens)

    
    xs = theano.shared(rng.randn(n_data, max_sequence_length, n_dim).astype(THEANOTYPE))
    masks = theano.shared(rng.randn(n_data, max_sequence_length).astype(THEANOTYPE))

    xs_numpy = xs.get_value()
    masks_numpy = masks.get_value()
    
    x1_lstms = lstm.BatchMultiLayerLSTM(
            rng, x1, m1, n_dim, n_hiddens=n_hiddens, output_type="last", prefix="lstms_x1")

    f1 = theano.function(
        inputs=[x1, m1],
        outputs=x1_lstms.output)

    small_n_data = 10
    sequence_lengths = [5, 10, 15, 20]
    xs0 = [rng.randn(small_n_data, n_dim).astype(THEANOTYPE)
           for n_data in sequence_lengths]
    xs_arr0, mask = lstm.batchify(xs0)

    
    f1_ind = theano.function(
        inputs=[x1_indices],
        outputs=x1_lstms.output,
        givens={
            x1: xs[x1_indices].swapaxes(0, 1),
            m1: masks[x1_indices]})
    
    fbatch = theano.function(
        inputs=[x1_indices, x2_indices, x3_indices],
        outputs=[model.x1_lstms.output, model.x2_lstms.output, model.x3_lstms.output],
        givens={
            x1: xs[x1_indices].swapaxes(0, 1),
            m1: masks[x1_indices],
            x2: xs[x2_indices].swapaxes(0, 1),
            m2: masks[x2_indices],
            x3: xs[x3_indices].swapaxes(0, 1),
            m3: masks[x3_indices],})
    ind1 = numpy.asarray([1, 2], dtype=numpy.int32)
    ind2 = numpy.asarray([1, 2], dtype=numpy.int32)
    ind3 = numpy.asarray([1, 2], dtype=numpy.int32)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""
Train a Siamese triplet CNN on the Switchboard data.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
from theano.tensor.shared_randomstreams import RandomStreams
import argparse
import cPickle as pickle
import logging
import numpy as np
import os
import scipy.spatial.distance as distance
import sys
import theano
from theano import tensor

sys.path.append(path.join("..", "..", "src", "couscous"))

import siamese
import training
import apply_layers
import data_io
import samediff

logger = logging.getLogger(__name__)
THEANOTYPE = theano.config.floatX

#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    "data_dir": "../data/nonpadded_icassp15.0",
    # "data_dir": "data/tmp",
    "n_same_pairs": int(100e3), # if None, all same pairs are used
    "n_hiddens": [80, 80],
    "rnd_seed": 42,
    "batch_size": 1024,
    "n_max_epochs": 20,
    "l1_weight": 0.0,
    "l2_weight": 0.0,
    "learning_rule": {
        "type": "adadelta",     # can be "momentum", "adadgrad", "adadelta"
        "rho": 0.9,             # parameters specific to learning rule
        "epsilon": 1e-6
        },
    # "learning_rule": {
    #     "type": "momentum",
    #     "learning_rate": 0.01,
    #     "momentum": 0.9
    #     },
    "dropout_rates": None,      # a list of rates for each layer or None
    "conv_layer_specs": [       # activation can be "sigmoid", "tanh", "relu", "linear"
        {"filter_shape": (96, 1, 39, 9), "pool_shape": (1, 3), "activation": "relu"},
        {"filter_shape": (96, 96, 1, 8), "pool_shape": (1, 3), "activation": "relu"},
        ],
    "hidden_layer_specs": [
        {"units": 2048, "activation": "relu"},
        {"units": 1024, "activation": "linear"}
        ],
    "loss": "hinge_cos",        # can be "hinge_cos"
    "margin": 0.15,              # margin for hinge loss
    }


#-----------------------------------------------------------------------------#
#                        TRAINING FUNCTIONS AND CLASSES                       #
#-----------------------------------------------------------------------------#

class TripletIterator(object):
    """In every epoch the tokens for the different-pairs are sampled."""
    
    def __init__(self, rng, matches_vec, sample_diff_every_epoch=True,
                 n_same_pairs=None):
        """
        If `n_same_pairs` is given, this number of same pairs is sampled,
        otherwise all same pairs are used.
        """
        self.rng = rng
        self.matches_vec = matches_vec

        self.same_matrix = distance.squareform(matches_vec)
        if n_same_pairs is None:
            # Use all pairs
            I, J = np.where(np.triu(self.same_matrix))  # indices of same pairs
        else:
            # Sample same pairs
            n_pairs = min(n_same_pairs, len(np.where(matches_vec == True)[0]))
            same_sample = self.rng.choice(
                np.where(matches_vec == True)[0], size=n_pairs, replace=False
                )
            same_vec = np.zeros(self.matches_vec.shape[0], dtype=np.bool)
            same_vec[same_sample] = True
            I, J = np.where(np.triu(distance.squareform(same_vec)))

        self.x1_same_indices = []
        self.x2_same_indices = []
        for i, j in zip(I, J):
            self.x1_same_indices.append(i)
            self.x2_same_indices.append(j)
            self.x1_same_indices.append(j)
            self.x2_same_indices.append(i)

        self.x1_same_indices = np.array(self.x1_same_indices, dtype=np.int32)
        self.x2_same_indices = np.array(self.x2_same_indices, dtype=np.int32)
        # self.x3_diff_indices = -1*np.ones(len(self.x1_same_indices), dtype=np.int32)

        np.fill_diagonal(self.same_matrix, True)
        if not sample_diff_every_epoch:
            self.x3_diff_indices = self._sample_diff_indices()
        self.sample_diff_every_epoch = sample_diff_every_epoch

        # assert not np.any(np.where(self.x1_same_indices == self.x3_diff_indices)[0])
        # assert not np.any(np.where(self.x2_same_indices == self.x3_diff_indices)[0])
        # assert not np.any(np.where(self.x1_same_indices == self.x2_same_indices)[0])

    def _sample_diff_indices(self):
        x3_diff_indices = -1*np.ones(len(self.x1_same_indices), dtype=np.int32)
        for i_token in xrange(self.same_matrix.shape[0]):
            cur_matches = np.where(np.array(self.x1_same_indices) == i_token)[0]
            if cur_matches.shape[0] > 0:
                x3_diff_indices[cur_matches] = self.rng.choice(
                    np.where(self.same_matrix[i_token] == False)[0],
                    size=len(cur_matches),
                    replace=True
                    )
                # if i_token == 32:
                #     print "!"
                #     print np.where(self.same_matrix[i_token] == True)[0]
                #     print x3_diff_indices[cur_matches]
        return x3_diff_indices

    def __iter__(self):

        # Sample different tokens for this epoch
        if self.sample_diff_every_epoch:
            x3_diff_indices = self._sample_diff_indices()
        else:
            x3_diff_indices = self.x3_diff_indices


        for ind in range(len(self.x1_same_indices)):
            batch_x1_ind = self.x1_same_indices[ind]
            batch_x2_ind = self.x2_same_indices[ind]
            batch_x3_ind = x3_diff_indices[ind]
            yield (batch_x1_ind, batch_x2_ind, batch_x3_ind)


def train_siamese_triplets_lstm(options_dict):
    """Train and save a Siamese CNN using the specified options."""

    # Preliminary

    logger.info(datetime.now())

    if not path.isdir(options_dict["model_dir"]):
        os.makedirs(options_dict["model_dir"])

    if "log_to_file" in options_dict and options_dict["log_to_file"] is True:
        log_fn = path.join(options_dict["model_dir"], "log")
        print "Writing:", log_fn
        root_logger = logging.getLogger()
        if len(root_logger.handlers) > 0:
            root_logger.removeHandler(root_logger.handlers[0])  # close open file handler
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.DEBUG)

    rng = np.random.RandomState(options_dict["rnd_seed"])
    if options_dict["dropout_rates"] is not None:
        srng = RandomStreams(seed=options_dict["rnd_seed"])
    else:
        srng = None

    options_dict_fn = path.join(options_dict["model_dir"], "options_dict.pkl.gz")
    logger.info("Saving options: " + options_dict_fn)
    f = data_io.smart_open(options_dict_fn, "wb")
    pickle.dump(options_dict, f, -1)
    f.close()

    logger.info("Options: " + str(options_dict))


    # Load and format data

    # Load into shared variables
    datasets = data_io.load_swbd_same_diff_nopadding(rng, options_dict["data_dir"])
    train_x, train_begins, train_ends, train_matches_vec, train_labels = datasets[0]
    dev_x, dev_begins, dev_ends, dev_matches_vec, dev_labels = datasets[1]
    test_x, test_begins, test_ends, test_matches_vec, test_labels = datasets[2]

    # Make batch iterators
    train_triplet_iterator = TripletIterator(
        rng, train_matches_vec, n_same_pairs=options_dict["n_same_pairs"],
        sample_diff_every_epoch=True)
    validate_triplet_iterator = TripletIterator(
        rng, dev_matches_vec, n_same_pairs=options_dict["n_same_pairs"],
        sample_diff_every_epoch=False)
    test_triplet_iterator = TripletIterator(
        rng, test_matches_vec, n_same_pairs=options_dict["n_same_pairs"],
        sample_diff_every_epoch=False)


    # Setup model

    logger.info("Building Siamese triplets LSTM")

    # Symbolic variables
    x1 = tensor.matrix("x1", dtype=THEANOTYPE)
    x2 = tensor.matrix("x2", dtype=THEANOTYPE)
    x3 = tensor.matrix("x3", dtype=THEANOTYPE)
    x1_indices = tensor.iscalar("x1_indices")
    x2_indices = tensor.iscalar("x2_indices")
    x3_indices = tensor.iscalar("x3_indices")

    # Build model
    input_shape = (options_dict["batch_size"], 1, 39, 200)
    model = siamese.SiameseTripletLSTM(
        rng, x1, x2, x3, n_in=39, n_hiddens=options_dict["n_hiddens"])
    if options_dict["loss"] == "hinge_cos":
        if options_dict["dropout_rates"] is not None:
            loss = model.dropout_loss_hinge_cos(options_dict["margin"])
        else:
            loss = model.loss_hinge_cos(options_dict["margin"])
        error = model.loss_hinge_cos(options_dict["margin"])  # doesn't include regularization or dropout
    else:
        assert False, "Invalid loss: " + options_dict["loss"]

    # Add regularization
    if options_dict["l1_weight"] > 0. or options_dict["l2_weight"] > 0.:
        loss = loss + options_dict["l1_weight"]*model.l1 + options_dict["l2_weight"]* model.l2

    # Compile test functions
    same_distance = model.cos_same()  # track the distances of same and different pairs separately
    diff_distance = model.cos_diff()
    outputs = [error, loss, same_distance, diff_distance]
    theano_mode = theano.Mode(linker="cvm")
    validate_model = theano.function(
        inputs=[x1_indices, x2_indices, x3_indices],
        outputs=outputs,
        givens={
            x1: dev_x[dev_begins[x1_indices]:dev_ends[x1_indices]],
            x2: dev_x[dev_begins[x2_indices]:dev_ends[x2_indices]],
            x3: dev_x[dev_begins[x3_indices]:dev_ends[x3_indices]],
            },
        mode=theano_mode,
        )
    test_model = theano.function(
        inputs=[x1_indices, x2_indices, x3_indices],
        outputs=outputs,
        givens={
            x1: test_x[test_begins[x1_indices]:test_ends[x1_indices]],
            x2: test_x[test_begins[x2_indices]:test_ends[x2_indices]],
            x3: test_x[test_begins[x3_indices]:test_ends[x3_indices]],
            },
        mode=theano_mode,
        )

    # Gradients and training updates
    parameters = model.parameters
    gradients = tensor.grad(loss, parameters)
    learning_rule = options_dict["learning_rule"]
    if learning_rule["type"] == "adadelta":
        updates = training.learning_rule_adadelta(
            parameters, gradients, learning_rule["rho"], learning_rule["epsilon"]
            )
    elif learning_rule["type"] == "momentum":
        updates = training.learning_rule_momentum(
            parameters, gradients, learning_rule["learning_rate"], learning_rule["momentum"]
            )
    else:
        assert False, "Invalid learning rule: " + learning_rule["type"]

    # Compile training function
    train_model = theano.function(
        inputs=[x1_indices, x2_indices, x3_indices],
        outputs=outputs,
        updates=updates,
        givens={
            x1: train_x[train_begins[x1_indices]:train_ends[x1_indices]],
            x2: train_x[train_begins[x2_indices]:train_ends[x2_indices]],
            x3: train_x[train_begins[x3_indices]:train_ends[x3_indices]],
            },
        mode=theano_mode,
        )


    # Train model

    logger.info("Training Siamese triplets CNN")
    record_dict_fn = path.join(options_dict["model_dir"], "record_dict.pkl.gz")
    record_dict = training.train_fixed_epochs_with_validation(
        options_dict["n_max_epochs"],
        train_model=train_model,
        train_triplet_iterator=train_triplet_iterator,
        validate_model=validate_model,
        validate_triplet_iterator=validate_triplet_iterator,
        test_model=test_model,
        test_triplet_iterator=test_triplet_iterator,
        save_model_func=model.save,
        save_model_fn=path.join(options_dict["model_dir"], "model.pkl.gz"),
        record_dict_fn=record_dict_fn,
        )


    # Extrinsic evaluation

    # Pass data trough model
    logger.info("Performing same-different evaluation")
    layers_output_dict = apply_layers.apply_layers(options_dict["model_dir"], "dev", batch_size=645)  # batch size covers 10965 out of 10966 tokens
    utt_ids = sorted(layers_output_dict.keys())
    embeddings = np.array([layers_output_dict[i] for i in utt_ids])
    labels = data_io.swbd_utts_to_labels(utt_ids)

    # Perform same-different
    distances = pdist(embeddings, metric="cosine")
    matches = samediff.generate_matches_array(labels)
    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False])
    logger.info("Validation average precision: " + str(ap))
    ap_fn = path.join(options_dict["model_dir"], "dev_ap.txt")
    with open(ap_fn, "w") as f:
        f.write(str(ap) + "\n")


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="directory to write the model to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def load_siamese_triplets_lstm(options_dict):

    model_fn = path.join(options_dict["model_dir"], "model.pkl.gz")

    # Symbolic variables
    x1 = tensor.matrix("x1", dtype=THEANOTYPE)
    x2 = tensor.matrix("x2", dtype=THEANOTYPE)
    x3 = tensor.matrix("x3", dtype=THEANOTYPE)

    # Random number generators
    rng = np.random.RandomState(options_dict["rnd_seed"])

    # Build model
    input_shape = (options_dict["batch_size"], 1, 39, 200)
    if options_dict.has_key("sequence_output_type"):
        sequence_output_type = options_dict["sequence_output_type"]
    else:
        sequence_output_type = "last"
    model = siamese.SiameseTripletLSTM(
        rng, x1, x2, x3, n_in=39, n_hiddens=options_dict["n_hiddens"], output_type=sequence_output_type)
    # Load saved parameters
    logger.info("Reading: " + model_fn)
    f = data_io.smart_open(model_fn)
    model.load(f)
    f.close()

    return model


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # logging.basicConfig(level=logging.DEBUG)

    # Set local options
    options_dict = default_options_dict.copy()
    options_dict["model_dir"] = args.model_dir

    # Train and save the model and options
    train_siamese_triplets_lstm(options_dict)


if __name__ == "__main__":
    main()



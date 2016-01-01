#!/usr/bin/env python

"""
Encode a same-different set using the layers from the given model.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import logging
import numpy
import sys
import theano
import theano.tensor as T

import data_io
import train_siamese_triplets_convlstm
# import train_cnn
# import train_mlp
# import train_siamese_cnn
# import train_siamese_triplets_cnn

logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_dir", type=str, help="model directory")
    parser.add_argument(
        "set", type=str, help="set to perform evaluation on", choices=["train", "dev", "test"]
        )
    parser.add_argument(
        "--batch_size", type=int, help="if not provided, a single batch is used"
        )
    parser.add_argument(
        "--i_layer", type=int, help="the layer of the network to use (default: %(default)s)", default=-1
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()



def apply_layers(model_dir, set, batch_size=None, i_layer=-1):

    logger.info(datetime.now())

    # Load the model options
    options_dict_fn = path.join(model_dir, "options_dict.pkl.gz")
    logger.info("Reading: " + options_dict_fn)
    f = data_io.smart_open(options_dict_fn)
    options_dict = pickle.load(f)
    # print options_dict
    f.close()

    # Load the dataset
    npz_fn = path.join(options_dict["data_dir"], "swbd." + set + ".npz")
    logger.info("Reading: " + npz_fn)
    npz = numpy.load(npz_fn)
    logger.info("Loaded " + str(len(npz.keys())) + " segments")

    model = train_siamese_triplets_convlstm.load_siamese_triplets_convlstm_minibatch(options_dict)
        

    # Load data into Theano shared variable
    utt_ids = sorted(npz.keys())
    xs = [npz[i] for i in utt_ids]
    ls = numpy.asarray([len(x) for x in xs], dtype=int)
    max_length = 200
    batch_size = options_dict["batch_size"]
    n_batches = (len(ls) - 1)/100 + 1
    blocked_size = n_batches * batch_size
    xs = numpy.zeros((blocked_size, max_length, npz[utt_ids[0]].shape[1]),
                      dtype=theano.config.floatX)
    mask = numpy.zeros((blocked_size, max_length), dtype=theano.config.floatX)
    for j, i in enumerate(utt_ids):
        xs[j][:ls[j]] = npz[i]
        mask[j][:ls[j]] = 1.0

    logger.info("Formatting into Theano shared variable")

    shared_x = theano.shared(xs, borrow=True)
    shared_mask = theano.shared(mask, borrow=True)

    # Compile function for passing segments through CNN layers
    x = model.input  # input to the tied layers
    x_i = T.lscalar()
    m = model.mask
    normalized_output = model.output
    apply_model = theano.function(
        inputs=[x_i],
        outputs=normalized_output,
        givens={
            x: shared_x[x_i*batch_size:batch_size*(x_i+1)],
            m: shared_mask[batch_size*x_i:batch_size*(x_i+1)].T
            }
        )

    logger.info(datetime.now())

    n_x = len(ls)
    logger.info("Passing data through in model: " + str(n_x))
    embeddings = []
    for x_i in range(1): # range(n_batches):
        x_embeddings = apply_model(x_i)
        embeddings.extend(x_embeddings)
    embeddings = numpy.vstack(embeddings[:len(ls)])
    logger.info("Outputs shape: " + str(embeddings.shape))

    embeddings_dict = {}

    for embedding_i, embedding in enumerate(embeddings):
        utt_id = utt_ids[embedding_i]
        embeddings_dict[utt_id] = embedding

    logger.info(datetime.now())

    return embeddings_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    logging.basicConfig(level=logging.DEBUG)

    layers_output_dict = apply_layers(args.model_dir, args.set, args.batch_size, args.i_layer)

    layers_output_npz_fn = path.join(
        args.model_dir, "swbd." + args.set + ".layer_" + str(args.i_layer) + ".npz"
        )
    logger.info("Writing: " + layers_output_npz_fn)
    numpy.savez_compressed(layers_output_npz_fn, **layers_output_dict)


if __name__ == "__main__":
    main()

# January 20, 2015

## Checking gradient clipping first is to just check the gradients

```bash
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_minibatch.1
```

# January 4, 2015

Main realization was that I was computing the features and using the data
wrong. The padding for the examples was actually on both sides and I learn my
lesson for not actually running through the padding function to see what was
going on.

Examining:
{'conv_layer_specs': [{'activation': 'relu', 'filter_shape': (96, 1, 39, 9), 'pool_shape': (1, 3)}, {'activation': 'relu\
', 'filter_shape': (96, 96, 1, 8), 'pool_shape': (1, 3)}], 'n_hiddens': [256, 256], 'rnd_seed': 42, 'l1_weight': 0.0, 'batch_size': 128, 'n_max\
_epochs': 20, 'dropout_rates': None, 'learning_rule': {'epsilon': 1e-06, 'type': 'adadelta', 'rho': 0.9}, 'l2_weight': 0.0, 'use_dropout_loss':\
 False, 'sequence_output_type': 'last', 'loss': 'hinge_cos', 'data_dir': '../data/icassp15.0', 'filter_shape': (96, 1, 9, 39), 'embedding_dim':\
 None, 'stabilize_activations': None, 'use_dropout_regularization': False, 'model_dir': '../models/siamese_triplets_convlstm.1', 'hidden_layer_\
specs':
[{'units': 2048, 'activation': 'relu'}, {'units': 1024, 'activation': 'linear'}],
'margin': 0.15, 'n_same_pairs': 100000}


# January 3, 2015

Adjusted dropout output so that things now can run with dropout and I'll see
whether that regularization works at all

```bash
python train_siamese_triplets_convlstm.py ../models/siamese_triplets_convlstm_dropout.1
python apply_layers_convlstm.py ../models/siamese_triplets_convlstm_dropout.1 dev
python eval_samediff.py ../models/siamese_triplets_convlstm_dropout.1/swbd.dev.layer_-1.npz 
```


# December 29, 2015

# testing things on MLPs

# December 28, 2015

Ensuring that I can the convolutional LSTM

```bash
mkdir -p ../models
python train_siamese_triplets_convlstm.py ../models/siamese_triplets_convlstm.1
python apply_layers_convlstm.py ../models/siamese_triplets_convlstm.1 dev
python eval_samediff.py ../models/siamese_triplets_convlstm.1/swbd.dev.layer_-1.npz 
```

```bash
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_last.1
python apply_layers.py ../models/siamese_triplets_lstm_last.1 dev
python eval_samediff.py ../models/siamese_triplets_lstm_last.1/swbd.dev.layer_-1.npz 
```
"dropout": .3
"sequence_output_type": "max"



# December 27, 2015

Made changes to the `lstm.py` file so that it now properly sets the output layer for multilayer lstms.

```bash
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_max_dropout.2
python apply_layers.py ../models/siamese_triplets_lstm_max_dropout.2 dev
python eval_samediff.py ../models/siamese_triplets_lstm_max_dropout.2/swbd.dev.layer_-1.npz 
```
"dropout": .3
"sequence_output_type": "max"


## without dropout

```bash
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_max.1
python apply_layers.py ../models/siamese_triplets_lstm_max.1 dev
python eval_samediff.py ../models/siamese_triplets_lstm_max.1/swbd.dev.layer_-1.npz 
```
"dropout": None
"sequence_output_type": "max"


# December 26, 2015

### This code will not work work anymore because I had a mistake
### and max_dropout.1 is actually last_dropout.1 as this has the
### last output type.
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_max_dropout.1
python apply_layers.py ../models/siamese_triplets_lstm_max_dropout.1 dev
python eval_samediff.py ../models/siamese_triplets_lstm_max_dropout.1/swbd.dev.layer_-1.npz 

Result was
```
Calculating average precision
Average precision: 0.00149335052629
Precision-recall breakeven: 0.0024533906916
2015-12-27 18:37:02.846835
```

# 

python train_mlp.py ../models/mlp.1
python apply_layers.py models/mlp.1 dev
python eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_minibatch.2
python apply_layers.py ../models/siamese_triplets_lstm_minibatch.2/ dev
python eval_samediff.py ../models/siamese_triplets_lstm_minibatch.2/swbd.dev.layer_-1.npz 


python train_lstm_mlp.py ../models/lstm_mlp.1
python lstm_mlp_apply_layers.py ../models/lstm_mlp.1 dev
python eval_samediff.py ../models/lstm_mlp.1/swbd.dev.layer_-1.npz

python train_lstm_mlp.py ../models/lstm_mlp.2
python lstm_mlp_apply_layers.py ../models/lstm_mlp.1 dev
python eval_samediff.py ../models/lstm_mlp.1/swbd.dev.layer_-1.npz

python lstm_mlp_apply_layers.py ../models/lstm_mlp.1 test
python eval_samediff.py ../models/lstm_mlp.1/swbd.test.layer_-1.npz

# lstm_mlp siamese triplets
python eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

# lstm nn

python train_siamese_triplet_lstm_nn.py ../models/siamese_triplet_lstm_nn.1/

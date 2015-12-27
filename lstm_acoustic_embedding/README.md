# December 27, 2015

Made changes to the `lstm.py` file so that it now properly sets the output layer for multilayer lstms.

```bash
python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_max_dropout.2
python apply_layers.py ../models/siamese_triplets_lstm_max_dropout.2 dev
python eval_samediff.py ../models/siamese_triplets_lstm_max_dropout.2/swbd.dev.layer_-1.npz 
```
"dropout": .3
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

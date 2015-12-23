python train_mlp.py ../models/mlp.1
python apply_layers.py models/mlp.1 dev
python eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

python siamese_triplets_lstm_minibatch.py ../models/siamese_triplets_lstm_minibatch.2
python apply_layers.py ../models/siamese_triplets_lstm_minibatch.2/ dev
python eval_samediff.py ../models/siamese_triplets_lstm_minibatch.2/swbd.dev.layer_-1.npz 


python train_lstm_mlp.py ../models/lstm_mlp.1
python lstm_mlp_apply_layers.py ../models/lstm_mlp.1 dev
python eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

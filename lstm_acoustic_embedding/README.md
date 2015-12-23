python train_lstm_mlp.py ../models/lstm_mlp.1
python lstm_mlp_apply_layers.py ../models/lstm_mlp.1 dev
python eval_samediff.py models/mlp.1/swbd.dev.layer_-1.npz

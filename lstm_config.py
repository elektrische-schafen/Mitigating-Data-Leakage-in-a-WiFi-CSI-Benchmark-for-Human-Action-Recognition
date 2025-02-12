from easydict import EasyDict as edict


def get_cfg():
    _C = edict()
    # Model parameters
    _C.input_size = 52    # Size of input features
    _C.hidden_size = 128  # Number of units in the LSTM layer
    _C.output_size = 7    # Number of output classes
    _C.seq_len = 300      # Sequence length
    _C.dropout_prob = 0.25
    _C.batch_size = 64
    _C.num_epochs = 50
    _C.train_portion = 0.75 # train/test ratio in case data split is carried out without respect to humans (_C.wrt_humans=False)
    _C.csi_har = "/mnt/data1/dvarga/Datasets/CSI-HAR-Dataset"
    _C.wrt_humans = False # True - data split with respect to humans; False - data split without respect to humans

    return _C

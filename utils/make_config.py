import configparser
import os

def make_config(model_path, data_path, anomaly_ratio=0.5, epoch=10, elayers=3, dff=512, input_c=38, batch_size=64, win_size=100, dmodel=512, patience=3, k=3, lr=0.0001):

    config = {}

    config['lr'] = 0.0001
    config['num_epoch'] = epoch
    config['k'] = k
    config['win_size'] = win_size
    config['input_c'] = input_c
    config['output_c'] = input_c
    config['batch_size'] = batch_size
    config['dataset'] = 'SMD'
    config['mode'] = 'test'
    config['data_path'] = data_path
    config['model_save_path'] = os.path.join('./Anomaly_transformer/result', model_path)
    config['anomaly_ratio'] = anomaly_ratio
    config['dmodel'] = dmodel
    config['dff'] = dff
    config['elayers'] = elayers
    config['patience'] = patience
    config['random_seed'] = 0

    return config
import configparser
import os

def make_config(model_path, data_path, anomaly_ratio=0.5, epoch=10, elayers=3, dff=512, input_c=38, batch_size=64, win_size=100, dmodel=512, patience=3, k=3, lr=0.0001):

    config = {}

    config['anomaly_ratio'] = anomaly_ratio
    config['epoch'] = epoch
    config['elayers'] = elayers
    config['dff'] = dff
    config['input_c'] = input_c
    config['output_c'] = input_c
    config['batch_size'] = batch_size
    config['win_size'] = win_size
    config['dmodel'] = dmodel
    config['patience'] = patience
    config['k'] = k

    config['lr'] = 0.001
    config['dataset'] = 'SMD'
    config['data_path'] = data_path
    config['mode'] = 'test'
    config['model_save_path'] = os.path.join('./Anomaly_transformer/result', model_path)

    return config
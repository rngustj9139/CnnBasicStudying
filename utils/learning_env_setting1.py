import os
import shutil

import numpy as np
import keras.models
import tensorflow as tf
from termcolor import colored
from keras.metrics import Mean, SparseCategoricalAccuracy

def dir_setting(dir_name, CONTINUE_LEARNING):
    cp_path = os.path.join(os.getcwd(), dir_name) # 현재 디렉터리 위치와 만들고자 하는 디렉터리 이름(dir_name)을 합침
    model_path = os.path.join(cp_path, 'model')

    if CONTINUE_LEARNING == False and os.path.isdir(cp_path): # CONTINUE_LEARNING이 False이고 디렉터리가 존재하면
        shutil.rmtree(cp_path)  # 디렉터리 내에 남아있는 정보 싹 지움
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True) # 디렉터리 만들기
        os.makedirs(model_path, exist_ok=True)

    path_dict = {
        'cp_path': cp_path,
        'model_path': model_path
    }

    return path_dict

def get_classification_metrics():
    train_loss = Mean()
    train_acc = SparseCategoricalAccuracy()

    validation_loss = Mean()
    validation_acc = SparseCategoricalAccuracy()

    test_loss = Mean()
    test_acc = SparseCategoricalAccuracy()

    metric_objects = dict()
    metric_objects['train_loss'] = train_loss
    metric_objects['train_acc'] = train_acc
    metric_objects['validation_loss'] = validation_loss
    metric_objects['validation_acc'] = validation_acc
    metric_objects['test_loss'] = test_loss
    metric_objects['test_acc'] = test_acc

    return metric_objects

def continue_setting(CONTINUE_LEARNING, path_dict, model=None): # 디폴트:None
    if CONTINUE_LEARNING == True: # 중간에 중단되었다가 다시 시작하는 경우
        epoch_list = os.listdir(path_dict['model_path'])
        epoch_list = [int(epoch.split('_')[1]) for epoch in epoch_list]
        epoch_list.sort()

        last_epoch = epoch_list[-1]
        model_path = path_dict['model_path'] + '/epoch_' + str(last_epoch)
        model = keras.models.load_model(model_path)

        losses_accs_path = path_dict['cp_path']
        losses_accs_np = np.load(losses_accs_path + '/losses_accs.npz')
        losses_accs = dict()

        for key, value in losses_accs_np.items():
            print(key, value)
            losses_accs[key] = list(value)

        start_epoch = last_epoch + 1
    else: # 처음부터 다시 시작하는 경우
        model = model
        start_epoch = 0
        losses_accs = {
            'train_losses':[], 'train_accs':[],
            'validation_losses':[], 'validation_accs':[],
            'test_losses':[], 'test_accs':[]
        }

    return model, losses_accs, start_epoch
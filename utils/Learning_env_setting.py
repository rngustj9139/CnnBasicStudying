import os
import shutil

from keras.metrics import Mean, SparseCategoricalAccuracy

def dir_setting(dir_name, CONTINUE_LEARNING):
    cp_path = os.path.join(os.getcwd(), dir_name)
    confusion_path = os.path.join(cp_path, 'confusion_matrix')
    model_path = os.path.join(cp_path, 'model')

    if CONTINUE_LEARNING == False and os.path.isdir(cp_path):
        shutil.rmtree(cp_path)  # 디렉터리 내에 남아있는 정보 싹 지움
    if not os.path.isdir(cp_path):
        os.makedirs(cp_path, exist_ok=True)
        os.makedirs(confusion_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    path_dict = {
        'cp_path': cp_path,
        'confusion_path': confusion_path,
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
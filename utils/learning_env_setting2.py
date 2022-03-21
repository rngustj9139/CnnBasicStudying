import os
from termcolor import colored
from utils.learning_env_setting1 import dir_setting, continue_setting

dir_name = 'train1'
CONTINUE_LEARNING = True

path_dict = dir_setting(dir_name, CONTINUE_LEARNING)

if CONTINUE_LEARNING == True and len(os.listdir(path_dict['model_path'])) == 0: # 예외처리
    CONTINUE_LEARNING = False
    print(colored('CONTINUE_LEARNING flag has been converted to FALSE', 'cyan'))

model = 'LeNet5()'

model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)


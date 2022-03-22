from termcolor import colored

def resetter(metric_objects): # 초기화
    metric_objects['train_loss'].reset_states()
    metric_objects['train_acc'].reset_states()
    metric_objects['validation_loss'].reset_states()
    metric_objects['validation_acc'].reset_states()

def training_reporter(epoch, losses_accs, metric_objects, exp_name=None): # 여러개의 모델을 시리얼하게 돌릴수도있기 때문에 모델의 이름을 나타내는 exp_name 변수 사용함, 모델마다 에포크를 달리할 수 있기 때문에 epoch 변수 사용함
    train_loss = metric_objects['train_loss']
    train_acc = metric_objects['train_acc']
    validation_loss = metric_objects['validation_loss']
    validation_acc = metric_objects['validation_acc']

    losses_accs['train_losses'].append(train_loss.result().numpy()) # 리스트에 추가
    losses_accs['train_accs'].append(train_acc.result().numpy() * 100)
    losses_accs['validation_losses'].append(validation_loss.result().numpy())
    losses_accs['validation_accs'].append(validation_acc.result().numpy() * 100)

    if exp_name:
        print(colored('Exp: ', 'red', 'on_white'), exp_name)
    print(colored('Epoch: ', 'red'), epoch)

    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}% \n' + \
        'Validation Loss: {:.4f}\t Validation Accuracy: {:.2f}% \n'
    print(template.format(losses_accs['train_losses'][-1], losses_accs['train_accs'][-1],
                          losses_accs['validation_losses'][-1], losses_accs['validation_accs'][-1]))
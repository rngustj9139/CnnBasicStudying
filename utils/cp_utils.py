import os
import numpy as np
import matplotlib.pyplot as plt

def save_metrics_model(epoch, model, losses_accs, path_dict, save_period):
    if epoch % save_period == 0:
        model.save(os.path.join(path_dict['model_path'], 'epoch_' + str(epoch)))

    np.savez_compressed(os.path.join(path_dict['cp_path'], 'losses_accs'),
                        train_losses=losses_accs['train_losses'],
                        train_accs=losses_accs['train_accs'],
                        validation_losses=losses_accs['validation_losses'],
                        validation_accs=losses_accs['validation_accs'])

def metric_visualizer(losses_accs, save_path): # 각 epoch마다 저장
    fig, ax = plt.subplots(figsize=(35, 15))
    ax2 = ax.twinx()

    epoch_range = np.arange(1, 1+len(losses_accs['train_losses']))

    ax.plot(epoch_range, losses_accs['train_losses'], color='tab:blue', linewidth=2, label='Train Loss')
    ax.plot(epoch_range, losses_accs['validation_losses'], color='tab:blue', linestyle=':', linewidth=2, label='Validation Loss')
    ax2.plot(epoch_range, losses_accs['train_accs'], color='tab:orange', linewidth=2, label='Train Accuracy')
    ax2.plot(epoch_range, losses_accs['validation_accs'], color='tab:orange', linestyle=':', linewidth=2, label='Validation Accuracy')

    ax.legend(bbox_to_anchor=(1, 0.5), loc='upper right', fontsize=20, frameon=False)
    ax2.legend(bbox_to_anchor=(1, 0.5), loc='lower right', fontsize=20, frameon=False)

    ax_yticks = ax.get_yticks()
    ax2_yticks = ax2.get_yticks()

    ax_yticks_m, ax_yticks_M = ax_yticks[0], ax_yticks[-1]

    ax_yticks = np.linspace(0, ax_yticks_M, 7)
    ax2_yticks = np.arange(20, 101, 5)
    ax2_yticks_minor = np.arange(20, 101, 1)

    ax.set_yticks(ax_yticks)
    ax.set_ylim([0, ax_yticks_M])
    ax.set_yticklabels(np.around(ax_yticks, 2))

    ax2_ylim = ax2.get_ylim()
    ax2.set_ylim([20, 100])
    ax2.set_yticks(ax2_yticks)
    ax2.set_yticks(ax2_yticks_minor, minor=True)

    epoch_ticks = np.linspace(1, len(losses_accs['train_losses']), 10).astype(np.int)

    ax.tick_params(labelsize=20, colors='tab:blue')
    ax2.tick_params(labelsize=20, colors='tab:orange')
    ax2.tick_params(which='minor', right=False)

    ax.set_xticks(epoch_ticks)
    ax2.set_xticks(epoch_ticks)
    ax.set_xticklabels(epoch_ticks, color='k')

    ax2.grid(axis='y')
    ax2.grid(which='minor', linestyle=':')

    ax.set_xlim(1, len(losses_accs['train_losses']))
    ax2.set_xlim(1, len(losses_accs['train_losses']))

    ax.set_ylabel('Cross Entropy Loss', fontsize=30, color='tab:blue')
    ax2.set_ylabel('Accuracy', fontsize=30, color='tab:orange')

    fig.savefig(save_path + '/losses_accs_visualization.png')
    plt.close()

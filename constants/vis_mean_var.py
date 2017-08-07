import pickle

with open('train_data_features_mean.pickle', 'rb') as f:
    train_mean = pickle.load(f)

with open('train_validate_data_features_mean.pickle', 'rb') as f:
    train_val_mean = pickle.load(f)
    
with open('all_data_features_mean.pickle', 'rb') as f:
    all_mean = pickle.load(f)

with open('train_data_features_var.pickle', 'rb') as f:
    train_var = pickle.load(f)

with open('train_validate_data_features_var.pickle', 'rb') as f:
    train_val_var = pickle.load(f)
    
with open('all_data_features_var.pickle', 'rb') as f:
    all_var = pickle.load(f)

import matplotlib.pylab as plt
import numpy as np

figure, ax = plt.subplots(1)

indices = np.arange(1152)

ax.scatter(indices, np.concatenate((train_mean['mean_rgb'], train_mean['mean_audio'])), s=1.4, label='mean')
# ax.scatter(indices, np.concatenate((train_val_mean['mean_rgb'], train_val_mean['mean_audio'])), label='train_val_mean')
# ax.scatter(indices, np.concatenate((all_mean['mean_rgb'], all_mean['mean_audio'])), label='all_mean')

ax.legend()
ax.set_xlabel('features (mean_rgb, mean_audio)')

ax.scatter(indices, np.concatenate((train_var['mean_rgb'], train_var['mean_audio'])), s=1.4, label='var')
# ax.scatter(indices, np.concatenate((train_val_var['mean_rgb'], train_val_var['mean_audio'])), label='train_val_var')
# ax.scatter(indices, np.concatenate((all_var['mean_rgb'], all_var['mean_audio'])), label='all_var')

ax.set_xlim([-2, 1154])
ax.set_xticks([1023.5])
ax.set_xticklabels([''])

ax.grid(b=True)
ax.legend(ncol=2)
ax.set_xlabel('features (mean_rgb, mean_audio)')


plt.show()

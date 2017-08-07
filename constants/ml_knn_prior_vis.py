import json, pickle, argparse
import matplotlib.pylab as plt
from os.path import join as path_join
import numpy as np

folder = '.'
k = 8

"""
def recover_prior_prob(folder=''):
    with open(folder + '/sum_labels.pickle', 'rb') as pickle_file:
        sum_labels = pickle.load(pickle_file)

    with open(folder + '/accum_num_videos.pickle', 'rb') as pickle_file:
        accum_num_videos = pickle.load(pickle_file)

    with open(folder + '/labels_prior_prob.pickle', 'rb') as pickle_file:
        labels_prior_prob = pickle.load(pickle_file)

    return sum_labels.tolist(), accum_num_videos.tolist(), labels_prior_prob.tolist()

sum_labels, accum_num_videos, labels_prior_prob = recover_prior_prob(folder=folder)

figure, (ax1, ax2) = plt.subplots(1, 2)
ax1.semilogy(sum_labels, color='#1f77b4', marker='.', markersize=0.1, label='Number of instances')
ax2.semilogy(labels_prior_prob, color='#ff7f0e', marker='.', markersize=0.1, label='Prior probability')
ax1.legend(loc=1)
ax2.legend(loc=1)

ax1.set_xlabel('Labels')
ax2.set_xlabel('Labels')

ax1.tick_params('y', colors='#1f77b4')
ax2.tick_params('y', colors='#ff7f0e')

plt.show()
"""

def restore_posterior_prob(k, folder=''):
    with open(path_join(folder, 'count_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            count = pickle.load(pickle_file)
        except:
            count = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'counter_count_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            counter_count = pickle.load(pickle_file)
        except:
            counter_count = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'pos_prob_positive_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            pos_prob_positive = pickle.load(pickle_file)
        except:
            pos_prob_positive = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'pos_prob_negative_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            pos_prob_negative = pickle.load(pickle_file)
        except:
            pos_prob_negative = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    return count, counter_count, pos_prob_positive, pos_prob_negative

count, counter_count, pos_prob_positive, pos_prob_negative = restore_posterior_prob(k, folder=folder)
"""
plt.imshow(pos_prob_positive, cmap='hot')
plt.show()
"""
"""
figure, (ax_t, ax_b) = plt.subplots(2, 1)

ind = np.arange(30)    # the x locations for the groups
ticks = np.arange(4716-30, 4716)
width = 0.35       # the width of the bars: can also be len(x) sequence

# Top
bar_list_t = []
for i, e in enumerate(pos_prob_positive[:, -30:]):
    bottom = None if i == 0 else np.sum(pos_prob_positive[:i, -30:], axis=0)
    p = ax_t.bar(left=ind, height=e, width=width, bottom=bottom)
    bar_list_t.append(p)

ax_t.set_title('Likelihoods versus label (positive class)')
ax_t.set_xlabel('Labels')
ax_t.set_ylabel('Likelihood')
ax_t.set_xlim([-1, 32])
ax_t.set_xticks(ind[::5])
ax_t.set_xticklabels(['{}'.format(i) for i in ticks[::5]])
ax_t.legend([p[0] for p in bar_list_t], ['{}'.format(j) for j in xrange(k+2)])

# Bottom
bar_list_b = []
for i, e in enumerate(pos_prob_negative[:, -30:]):
    bottom = None if i == 0 else np.sum(pos_prob_negative[:i, -30:], axis=0)
    p = ax_b.bar(left=ind, height=e, width=width, bottom=bottom)
    bar_list_b.append(p)

ax_b.set_title('Likelihoods versus label (negative class)')
ax_b.set_xlabel('Labels')
ax_b.set_ylabel('Likelihood')
ax_b.set_xlim([-1, 32])
ax_b.set_xticks(ind[::5])
ax_b.set_xticklabels(['{}'.format(i) for i in ticks[::5]])
ax_b.legend([p[0] for p in bar_list_t], ['{}'.format(j) for j in xrange(k+2)])

figure.tight_layout()
plt.show()
"""
"""
figure, (ax_t, ax_b) = plt.subplots(2, 1)

ticks = np.arange(0, 50)
ind = np.arange(len(ticks))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

# Top
bar_list_t = []
for i, e in enumerate(pos_prob_positive[:, ticks]):
    bottom = None if i == 0 else np.sum(pos_prob_positive[:i, ticks], axis=0)
    p = ax_t.bar(left=ind, height=e, width=width, bottom=bottom)
    bar_list_t.append(p)

ax_t.set_title('Likelihoods versus label (positive class)')
# ax_t.set_xlabel('Labels')
ax_t.set_ylabel('Likelihood')
ax_t.set_xlim([-1, 50])
ax_t.set_xticks(ind[::5].tolist() + [ind[-1]])
ax_t.set_xticklabels(['{}'.format(i) for i in ticks[::5]] + ['Labels'])
ax_t.legend([p[0] for p in bar_list_t], ['{}'.format(j) for j in xrange(k+2)], ncol=(k+1),
            bbox_to_anchor=(0., -0.25, 1., .102), loc=3, mode='expand', borderaxespad=0.)

# Bottom
bar_list_b = []
for i, e in enumerate(pos_prob_negative[:, ticks]):
    bottom = None if i == 0 else np.sum(pos_prob_negative[:i, ticks], axis=0)
    p = ax_b.bar(left=ind, height=e, width=width, bottom=bottom)
    bar_list_b.append(p)

ax_b.set_title('Likelihoods versus label (negative class)')
# ax_b.set_xlabel('Labels')
ax_b.set_ylabel('Likelihood')
ax_b.set_xlim([-1, 50])
ax_b.set_xticks(ind[::5].tolist() + [ind[-1]])
ax_b.set_xticklabels(['{}'.format(i) for i in ticks[::5]] + ['Labels'])
# ax_b.legend([p[0] for p in bar_list_t], ['{}'.format(j) for j in xrange(k+2)], ncol=(k+3)/2, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            mode='expand', borderaxespad=0.)

figure.tight_layout()

plt.show()
"""

# posterior probability
def recover_prior_prob(folder=''):
    with open(folder + '/sum_labels.pickle', 'rb') as pickle_file:
        sum_labels = pickle.load(pickle_file)

    with open(folder + '/accum_num_videos.pickle', 'rb') as pickle_file:
        accum_num_videos = pickle.load(pickle_file)

    with open(folder + '/labels_prior_prob.pickle', 'rb') as pickle_file:
        labels_prior_prob = pickle.load(pickle_file)

    return sum_labels.tolist(), accum_num_videos.tolist(), labels_prior_prob.tolist()

sum_labels, accum_num_videos, labels_prior_prob = recover_prior_prob(folder=folder)
labels_prior_prob = np.array(labels_prior_prob)

positive_prob_numerator = np.multiply(labels_prior_prob, pos_prob_positive)
negative_prob_numerator = np.multiply(1.0 - labels_prior_prob, pos_prob_negative)

posterior_prob = np.true_divide(positive_prob_numerator, positive_prob_numerator + negative_prob_numerator)

figure, ax = plt.subplots(1)

ticks = np.arange(0, 4716, 100)
ind = np.arange(len(ticks))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

# Top
bar_list = []
for i, e in enumerate(posterior_prob[:, ticks]):
    bottom = None if i == 0 else np.sum(posterior_prob[:i, ticks], axis=0)
    p = ax.bar(left=ind, height=e, width=width, bottom=bottom)
    bar_list.append(p)

ax.set_title('Posterior probability for an instance to be in positive class versus label')
# ax.set_xlabel('Labels')
ax.set_ylabel('Probability')
ax.set_xlim([-1, ind[-1]+1])
ax.set_xticks(ind[::5].tolist() + [ind[-1]])
ax.set_xticklabels(['{}'.format(i) for i in ticks[::5]] + ['Labels'])
ax.legend([p[0] for p in bar_list], ['{}'.format(j) for j in xrange(k+2)], ncol=(k+3)/2, loc=1)

figure.tight_layout()

plt.show()
"""
"""
print('Done')

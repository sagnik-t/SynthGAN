import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import metrics

class MANOVA(metrics.Metric):
    def __init__(self, name='manova', **kwargs):
        super(MANOVA, self).__init__(name=name, **kwargs)
        self.sum_within = self.add_weight(name='sum_within', shape=(None, None), initializer='zeros', aggregation='sum')
        self.sum_between = self.add_weight(name='sum_between', shape=(None, None), initializer='zeros', aggregation='sum')
        self.sample_count = self.add_weight(name='sample_count', initializer='zeros')
        self.group_means = []
        self.groups = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        group_labels = y_true.numpy()
        data = y_pred.numpy()

        unique_groups = np.unique(group_labels)
        overall_mean = np.mean(data, axis=0)
        group_means = {group: np.mean(data[group_labels == group], axis=0) for group in unique_groups}
        
        self.group_means.append(group_means)
        self.groups.append(unique_groups)

        for group in unique_groups:
            group_data = data[group_labels == group]
            group_mean = group_means[group]
            within_group_diff = group_data - group_mean
            self.sum_within.assign_add(tf.linalg.matmul(within_group_diff.T, within_group_diff))
            self.sum_between.assign_add(
                tf.linalg.matmul(
                    (group_mean - overall_mean).reshape(-1, 1),
                    (group_mean - overall_mean).reshape(1, -1)
                ) * len(group_data)
            )

        self.sample_count.assign_add(len(data))

    def result(self):
        total_sum = self.sum_within + self.sum_between
        wilks_lambda = tf.linalg.det(self.sum_within) / tf.linalg.det(total_sum)
        return wilks_lambda

    def reset_states(self):
        self.sum_within.assign(tf.zeros_like(self.sum_within))
        self.sum_between.assign(tf.zeros_like(self.sum_between))
        self.sample_count.assign(0)
        self.group_means = []
        self.groups = []
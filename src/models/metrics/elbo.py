import tensorflow as tf
import keras
from keras import losses

@keras.saving.register_keras_serializable(package='Custom', name='ELBOLoss')
class ELBOLoss(losses.Loss):
    
    def __init__(self, **kwargs):
        super(ELBOLoss, self).__init__(**kwargs)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        reconstruction_loss = losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
        kl_loss = losses.kl_divergence(y_true=y_true, y_pred=y_pred)
        return tf.reduce_mean(reconstruction_loss + kl_loss)
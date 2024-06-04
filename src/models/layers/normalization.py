import tensorflow as tf
import keras
from keras import Layer, layers

@keras.utils.register_keras_serializable(package='Custom', name='MinMaxNormalization')
class MinMaxNormalization(Layer):
    
    def __init__(self, min_val, max_val, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        scalar = lambda x: ((x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))) * (max_val - min_val) + min_val
        self.lambda_layer = layers.Lambda(scalar)
        
    def call(self, inputs):
        return self.lambda_layer(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'min_val': self.min_val,
            'max_val': self.max_val,
        })
        return config
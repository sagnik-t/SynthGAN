import tensorflow as tf
from keras import Layer, layers

class Split(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=1)
        user_id, item_id, rating = tf.split(inputs, num_or_size_splits=3, axis=1)
        layers.Reshape(target_shape=(1,))(user_id)
        layers.Reshape(target_shape=(1,))(item_id)
        layers.Reshape(target_shape=(1,))(rating)
        return user_id, item_id, rating
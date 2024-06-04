import tensorflow as tf
import keras
from keras import Layer, layers

@keras.utils.register_keras_serializable(package='Custom', name='Split')
class Split(Layer):
    
    def __init__(self, num_splits, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.num_splits = num_splits
        self.reshape = layers.Reshape(target_shape=(1,))
    
    def call(self, inputs) -> list[tf.Tensor]:
        if isinstance(inputs, (list, tuple)):
            inputs = tf.stack(inputs, axis=1)
        
        splits = tf.split(value=inputs, num_or_size_splits=self.num_splits, axis=1)
        
        return [self.reshape(split) for split in splits]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_splits': self.num_splits,
        })
        return config
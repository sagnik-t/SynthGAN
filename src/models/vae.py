from typing import Literal

import tensorflow as tf
import keras
from keras import Model, layers, losses, optimizers

@keras.saving.register_keras_serializable(package='Custom', name='VAE')
class VAE(Model):
    
    def __init__(
        self,
        encoder: Model,
        decoder: Model,
        latent_dim: int,
        **kwargs
    ):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
    
    def call(self,inputs: tf.Tensor, training=False):
        encoded = self.encoder.call(inputs, training=training)
        decoded = self.decoder.call(encoded, training=training)
        return decoded

    def predict(self, x: tf.Tensor, mode: Literal['encode', 'decode']) -> tf.Tensor:
        if mode == 'encode':
            return self.encoder.predict(x)
        else:
            return self.decoder.predict(x)
        
    def build_graph(self):
        if isinstance(self.encoder, keras.src.models.Functional):
            input_shape = self.encoder.layers[1].input.shape[1:]
        else:
            input_shape = self.encoder.layers[0].input.shape[0:]
        
        input_tensor = layers.Input(shape=input_shape)
        
        return Model(inputs=input_tensor, outputs=self.call(input_tensor))
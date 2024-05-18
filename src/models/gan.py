import tensorflow as tf
import keras
from keras import Model, layers, losses


class GAN(Model):
    
    def __init__(
        self,
        generator: Model,
        discrimator: Model,
        latent_dim: int = 100,
        **kwargs
    ):
        super(GAN, self).__init__(**kwargs)
        self.generator = generator
        self.discrimator = discrimator
        self.latent_dim = latent_dim
    
    def call(self, inputs, training=False):
        noise_vec, input_tensor = inputs
        
        gen_tensor = self.generator(noise_vec, training=True)
        
        real_output = self.discrimator(input_tensor, training=True)
        gen_output = self.discrimator(gen_tensor, training=True)
        
        return gen_tensor, real_output, gen_output
    
    def compile(
        self,
        gen_optimizer: keras.Optimizer,
        disc_optimizer: keras.Optimizer,
        **kwargs
    ):
        super(GAN, self).compile(**kwargs)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        _, real_output, gen_output = y_pred
        
        gen_loss = losses.binary_crossentropy(tf.ones_like(gen_output), gen_output)
        disc_loss = losses.binary_crossentropy(tf.ones_like(real_output), real_output) + losses.binary_crossentropy(tf.zeros_like(gen_output), gen_output)
        
        return {
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        }
    
    def train_step(self, inputs):
        
        noise_vec = tf.random.normal((tf.shape(inputs)[0], self.latent_dim))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            _, real_output, gen_output = self.call(inputs=(noise_vec, inputs), training=True)
            loss = self.compute_loss(y=[inputs, real_output, gen_output])
        
        gen_grads = gen_tape.gradient(loss['gen_loss'], self.generator.trainable_weights)
        disc_grads = disc_tape.gradient(loss['disc_loss'], self.discrimator.trainable_weights)
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discrimator.trainable_weights))
        
        return {
            'gen_loss': loss['gen_loss'],
            'disc_loss': loss['disc_loss']
        }
    
    def predict(self, x):
        return self.generator(x)
    
    def summary(self):
        model = self.build_graph()
        return model.summary()
    
    def build_graph(self):
        if isinstance(self.discrimator, keras.src.models.Functional):
            input_shape = self.discrimator.layers[1].input.shape[1:]
        else:
            input_shape = self.discrimator.layers[0].input.shape[1:]
        
        input_tensor = layers.Input(shape=input_shape)
        noise_vec = layers.Input(shape=(self.latent_dim,))
        
        return Model(inputs=[noise_vec, input_tensor], outputs=self.call([noise_vec, input_tensor]))
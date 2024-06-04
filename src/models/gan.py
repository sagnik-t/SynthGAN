import tensorflow as tf
import keras
from keras import Model, layers, losses

@keras.utils.register_keras_serializable(package='Custom', name='GAN')
class GAN(Model):
    
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        noise_dim: int = 100,
        **kwargs
    ):
        super(GAN, self).__init__(**kwargs)
        self.noise_dim = noise_dim
        self.generator = generator
        self.discriminator = discriminator
    
    def call(self, inputs, training=False) -> tuple[tf.Tensor]:
        noise_vec, input_tensor = inputs
        
        gen_tensor = self.generator(noise_vec, training=True)
        
        real_output = self.discriminator(input_tensor, training=True)
        gen_output = self.discriminator(gen_tensor, training=True)
        
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
        
        noise_vec = tf.random.normal((tf.shape(inputs)[0], self.noise_dim))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_tensor, real_output, gen_output = self.call(inputs=(noise_vec, inputs), training=True)
            loss = self.compute_loss(y_pred=[inputs, real_output, gen_output])
        
        gen_grads = gen_tape.gradient(loss['gen_loss'], self.generator.trainable_weights)
        disc_grads = disc_tape.gradient(loss['disc_loss'], self.discriminator.trainable_weights)
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_weights))
        
        return {
            'gen_loss': loss['gen_loss'],
            'disc_loss': loss['disc_loss'],
            **self.compute_metrics(x=noise_vec, y=inputs, y_pred=gen_tensor)
        }
    
    def predict(self, x):
        return self.generator(x)
    
    def summary(self, **kwargs):
        model = self.build_graph()
        return model.summary(**kwargs)
    
    def build_graph(self):
        if isinstance(self.discriminator, keras.src.models.Functional):
            input_shape = self.discriminator.layers[1].input.shape[1:]
        else:
            input_shape = self.discriminator.layers[0].input.shape[0:]
        
        input_tensor = layers.Input(shape=input_shape)
        noise_vec = layers.Input(shape=(self.noise_dim,))
        
        return Model(inputs=[noise_vec, input_tensor], outputs=self.call((noise_vec, input_tensor)))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_dim': self.noise_dim,
            'generator': self.generator.get_config(),
            'discriminator': self.discriminator.get_config()
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        generator = Model.from_config(config.pop('generator'))
        discriminator = Model.from_config(config.pop('discriminator'))
        return cls(generator=generator, discriminator=discriminator, **config)
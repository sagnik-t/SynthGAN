import tensorflow as tf
import keras
from keras import optimizers, losses, metrics

from config import Config
from models import GAN
from pipe import FnArgs

def run_gan(fn_args: FnArgs):
    
    gan = GAN(
        generator=fn_args.generator,
        discriminator=fn_args.discriminator,
        noise_dim=fn_args.noise_dim
    )
    
    gan.compile(
        gen_optimizer=optimizers.Adam(),
        disc_optimizer=optimizers.Adam()
    )
    
    gan.fit(
        x=fn_args.train_set[0],
        epochs=fn_args.epochs,
        batch_size=fn_args.batch_size
    )
    
    gan.save(filepath=Config.Paths.REGISTRY_PATH / 'gan.keras')
    
    keras.utils.plot_model(
        gan.build_graph(),
        show_shapes=True,
        expand_nested=True,
        dpi=70,
        to_file=Config.Paths.IMAGE_PATH / 'gan.png'
    )
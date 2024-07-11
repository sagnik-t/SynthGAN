import tensorflow as tf
import keras
from keras import optimizers, losses, metrics

from config import Config
from models import VAE
from pipe import FnArgs

def run_vae(fn_args: FnArgs):
    
    vae = VAE(
        generator=fn_args.encoder,
        discriminator=fn_args.decoder,
        latent_dim=fn_args.latent_dim
    )
    
    vae.compile(
        optimizer=optimizers.Adam(),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.R2Score()]
    )
    
    vae.fit(
        x=fn_args.train_set[0],
        epochs=fn_args.epochs,
        batch_size=fn_args.batch_size
    )
    
    vae.save(filepath=Config.Paths.REGISTRY_PATH / 'vae.keras')
    
    keras.utils.plot_model(
        vae.build_graph(),
        show_shapes=True,
        expand_nested=True,
        dpi=70,
        to_file=Config.Paths.IMAGE_PATH / 'vae.png'
    )
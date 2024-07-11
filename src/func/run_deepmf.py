import tensorflow as tf
import keras
from keras import optimizers, losses, metrics

from config import Config
from models import DeepMF
from pipe import FnArgs 

def run_deepmf(fn_args: FnArgs):
    
    deepmf = DeepMF()
    
    deepmf.compile(
        optimizer=optimizers.Adam(),
        loss=losses.MeanSquaredError(),
        metrics=[metrics.R2Score()]
    )
    
    deepmf.fit(
        x=fn_args.train_set[0],
        y=fn_args.train_set[1],
        validation_data=(
            fn_args.val_set[0],
            fn_args.val_set[1]
        ),
        epochs=fn_args.epochs,
        batch_size=fn_args.batch_size,
        validation_freq=fn_args.validation_freq
    )
    
    deepmf.save(filepath=Config.Paths.REGISTRY_PATH / 'deepmf.keras')
    
    keras.utils.plot_model(
        deepmf.build_graph(),
        show_shapes=True,
        dpi=70,
        to_file=Config.Paths.IMAGE_PATH / 'deepmf.png'
    )
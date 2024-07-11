from typing import Callable, Dict, Any

import tensorflow as tf
import keras

from config import Config
from pipe import FnArgs

class Transform:
    
    def __init__(
        self,
        name: str,
        run_fn: Callable,
        params: Dict[str, Any] = None
    ):
        self.name = name
        self.run_fn = run_fn
        self.params = params
        
        self.path = Config.Paths.REGISTRY_PATH.joinpath(self.name + '.keras')
        self._ran = True if Config.Paths.REGISTRY_PATH.joinpath(self.name).exists() else False
    
    def run(
        self,
        pipe_args: FnArgs = None
    ) -> None:
        fn_args = pipe_args.update(self.params)
        self.run_fn(fn_args)
        self._ran = True
    
    def transform(self, data: tf.Tensor) -> tf.Tensor:
        model = keras.models.load_model(filepath=Config.Paths.REGISTRY_PATH.joinpath(self.name + '.keras'))
        return model.predict(data)
    
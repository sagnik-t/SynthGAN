from typing import List, Tuple, Dict, Any
from pathlib import Path
from numpy.typing import ArrayLike

import tensorflow as tf

from config import Config
from pipe import FnArgs, Trainer, Transform

class TrainingPipeline:
    
    def __init__(
        self,
        components: List[Trainer | Transform],
        train_set: Tuple[ArrayLike, ArrayLike | None],
        val_set: Tuple[ArrayLike, ArrayLike | None] | None = None,
        pipe_args: Dict[str, Any] | None = None,
        resume: bool = False,
    ):
        self.components = components
        self.pipe_args = FnArgs(**{
            'train_set': train_set,
            'val_set': val_set,
            'pipe_args': pipe_args
        })
        self.resume = resume
        self._ran = True if self.run_check() else False
    
    def run(self) -> None:
        
        if not self.resume:
            # TrainingPipeline.clean_registry()
            print("Cleaning registry")
        
        for component in self.components:
            
            if not self.resume or (self.resume and not component._ran):
                component.run(pipe_args=self.pipe_args)
                
            if isinstance(component, Transform):
                self.pipe_args.train_set = (
                    component.transform(
                        data=self.pipe_args.train_set[0]
                    ),
                    self.pipe_args.train_set[1]
                )
                
        self._ran = True
        return
    
    def infer(self, data: ArrayLike) -> tf.Tensor:
        
        if not self._ran:
            raise PipelineNotRunError()
        
        return self.components[-1].infer(data)
    
    def get_component(self, name: str) -> Trainer | Transform:
        for component in self.components:
            if component.name == name:
                return component
        return None

    def run_check(self):
        for component in self.components:
            if not component._ran:
                return False
        return True
    
    @staticmethod
    def clean_registry() -> None:
        for file in Config.Paths.REGISTRY_PATH.iterdir():
            if file.is_file():
                file.unlink()


class PipelineNotRunError(Exception):
    
    def __init__(self, msg="Pipeline has not been run yet"):
        super().__init__(msg)
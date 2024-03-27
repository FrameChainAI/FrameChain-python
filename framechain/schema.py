from enum import Enum
from typing import Callable, Literal, Optional, Self
from framechain.chains.functional import FunctionalChain
from abc import ABC, abstractmethod

import stringcase
import numpy as np
from pydantic import BaseModel

from framechain.utils.image_type import ImageType


class Serializable(ABC, BaseModel):
    type_id: str  # should be unique for each model and remain constant
    version: str  # should be updated when the model changes
    meta: dict  # should contain any additional information about the model schema itself (not instance data)

    @abstractmethod
    def serialize(self) -> str:
        pass

    @abstractmethod
    @classmethod
    def deserialize(self, text: str) -> Self:
        pass


RunInput = dict
RunOutput = dict


class Runnable(ABC):

    def run(self, **inputs: RunInput) -> RunOutput | None:

        possible_new_inputs = self.pre_run(inputs=inputs)
        if possible_new_inputs is not None:
            inputs = possible_new_inputs

        outputs = self._run(inputs=inputs)

        possible_new_outputs = self.post_run(inputs=inputs, outputs=outputs)
        if possible_new_outputs is not None:
            outputs = possible_new_outputs

        return outputs

    @abstractmethod
    def pre_run(self, inputs: RunInput | None) -> RunInput | None:
        """Called before the main _run method. Good place for logging, validation, etc."""
        return inputs

    @abstractmethod
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        """The main method that does the work. Should be overridden by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _run")

    @abstractmethod
    def post_run(
        self, inputs: RunInput | None, outputs: RunOutput | None
    ) -> RunOutput | None:
        """Called after the main _run method. Good place for logging, validation, etc."""
        return outputs

    def __call__(self, inputs: RunInput | None) -> RunOutput | None:
        return self.run(**inputs)

    def __or__(self, other):
        return SequentialRunnables(self, other)

    def __and__(self, other):
        return ParallelRunnables(self, other)


class CompositeRunnable(Runnable):
    runnables: list[Runnable]

    def __init__(self, *runnables: list[Runnable], **kwargs):
        if "runnables" in kwargs:
            runnables = kwargs["runnables"]
        super().__init__(runnables=runnables, **kwargs)


class SequentialRunnables(CompositeRunnable):

    def _run(self, **kwargs) -> dict | None:
        for runnable in self.runnables:
            kwargs = runnable(**kwargs)
        return kwargs

    def __or__(self, other):
        self.runnables.append(other)
        return self


class ParallelRunnables(CompositeRunnable):

    def _run(self, **kwargs) -> dict | None:
        new_kwargs = {}
        for runnable in self.runnables:
            updates = runnable(**kwargs)
            new_kwargs.update(updates)
            # FIXME: make this actually parallel when i implement async support
        return new_kwargs

    def __and__(self, other):
        self.runnables.append(other)
        return self


class BaseChain(Runnable, Serializable, ABC):

    inputs: list[str]
    outputs: list[str]
    

    @classmethod
    def from_func(cls, **kwargs):
        def dec(func: Callable):
            name = kwargs.get('name', stringcase.camelcase(f"{func.__name__}{cls.__name__}"))
            bases = kwargs.get('bases', (FunctionalChain, cls))
            return type(name, bases, {"func": func, **kwargs})
        return dec


class BaseImageChain(BaseChain, ABC):
    input_channels: Optional[int] = None
    input_type: Optional[ImageType] = None
    output_channels: Optional[int] = None
    output_type: Optional[ImageType] = None

    def pre_run(self, inputs: RunInput) -> RunInput:
        processed_inputs = self._process_io(
            image=inputs[self.input_image],
            channel_format=self.input_channels,
            image_type=self.input_type
        )
        inputs.update(processed_inputs)
        return super().pre_run(inputs)

    def post_run(self, inputs: RunInput, outputs: RunOutput) -> RunOutput:
        processed_outputs = self._process_io(
            image=outputs[self.output_image],
            channel_format=self.standard_output_channels,
            image_type=self.standard_output_format
        )
        outputs.update(processed_outputs)
        return super().post_run(inputs, outputs)
        
    def _process_io(self, *, image: Image, channel_format: int, image_type: str) -> dict:
        for k, v in image.items():
            if channel_format is not None:
                v = convert_channel_format(v, channel_format)
            if image_type is not None:
                v = convert_type(v, image_type)
            image[k] = v
        return image

class ModelBase(BaseChain, ABC):
    pass

class ImageModel(ModelBase, ABC):
    model_name: str
    model_type: str

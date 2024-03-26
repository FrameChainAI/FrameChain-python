from enum import Enum
from typing import Callable, Literal
import PIL
import numpy as np
from framechain.transformations.scale import ScalingMode, scale
import hashlib

from framechain.utils import standardize_images_in_dict

type Size = tuple[int, int] | tuple[float, float]
type Image = PIL.Image | np.ndarray
type ImageSeq = list[Image]


class ImageFormat(Enum):
    PIL = "PIL"
    np = "np"


def to(image, format: ImageFormat):
    if format == ImageFormat.PIL:
        return image if isinstance(image, PIL.Image) else PIL.Image.fromarray(image)
    if format == ImageFormat.np:
        return image if isinstance(image, np.ndarray) else np.array(image)


from abc import ABC, abstractmethod
from typing import Literal, Optional, Self

import numpy as np

from abc import ABC

from pydantic import BaseModel


class Serializable(ABC, BaseModel):
    type_id: str  # should be unique for each model and remain constant
    version: str  # should be updated when the model changes
    meta: dict  # should contain any additional information about the model

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
        pass

    @abstractmethod
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        """The main method that does the work. Should be overridden by subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement _run")

    @abstractmethod
    def post_run(
        self, inputs: RunInput | None, outputs: RunOutput | None
    ) -> RunOutput | None:
        """Called after the main _run method. Good place for logging, validation, etc."""
        pass

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


class Chain(Runnable, Serializable, ABC):

    min_input_size: Optional[Size] = None
    max_input_size: Optional[Size] = None
    preferred_input_size: Optional[Size] = None
    scale_inputs: bool = False
    standardize_input_channels: bool = False
    standard_input_channels: Optional[int] = 3
    input_scaling_mode: ScalingMode
    standardize_input_format: bool = True
    standard_input_format: ImageFormat = ImageFormat.np

    min_output_size: Optional[Size] = None
    max_output_size: Optional[Size] = None
    preferred_output_size: Optional[Size] = None
    scale_outputs: bool = False
    standardize_output_channels: bool = False
    standard_output_channels: Optional[int] = 3
    output_scaling_mode: ScalingMode
    standardize_output_format: bool = True
    standard_output_format: ImageFormat = ImageFormat.np
    
    def pre_run(self, **inputs: RunInput) -> RunInput:
        return self._process_io(
            io=inputs,
            standardize_channels=self.standardize_input_channels,
            standardize_format=self.standardize_input_format,
            scale=self.scale_inputs,
            min_size=self.min_input_size,
            max_size=self.max_input_size,
            preferred_size=self.preferred_input_size,
            scaling_mode=self.input_scaling_mode,
            standard_channels=self.standard_input_channels,
            standard_format=self.standard_input_format
        )

    def post_run(self, **outputs: RunOutput) -> RunOutput:
        return self._process_io(
            io=outputs,
            standardize_channels=self.standardize_output_channels,
            standardize_format=self.standardize_output_format,
            scale=self.scale_outputs,
            min_size=self.min_output_size,
            max_size=self.max_output_size,
            preferred_size=self.preferred_output_size,
            scaling_mode=self.output_scaling_mode,
            standard_channels=self.standard_output_channels,
            standard_format=self.standard_output_format
        )
    @classmethod
    def from_func(cls, func: Callable, /, **kwargs):
        return type(
            f"{func.__name__}{cls.__name__}",
            (FunctionalChain, cls),
            {"func": func, **kwargs},
        )

    def _process_io(self, *, io: dict, standardize_channels: bool, standardize_format: bool,
                    scale: bool, min_size: int, max_size: int, preferred_size: int,
                    scaling_mode: str, standard_channels: int, standard_format: str) -> dict:
        for k, v in io.items():
            if isinstance(v, PIL.Image):
                if standardize_channels:
                    v = v.convert("L" if standard_channels == 1 else "RGB") # FIXME: does this work?
                if standardize_format:
                    v = to(v, standard_format)
                if scale:
                    v = scale(
                        v,
                        min_size=min_size,
                        max_size=max_size,
                        preferred_size=preferred_size,
                        scaling_mode=scaling_mode
                    )
                io[k] = v
        return io


class FunctionalChain(Chain):
    func: Callable

    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        return self.func(**inputs)

    def serialize(self) -> str:
        func_code = self.func.__code__.co_code
        return hashlib.sha256(func_code).hexdigest()


def chain(base_chain: type[Chain] = Chain, /, **kwargs):
    def dec(func: Callable):
        return base_chain.from_func(func, **kwargs)

    return dec


class ImageModel(Chain, ABC):
    model_name: str
    model_type: str

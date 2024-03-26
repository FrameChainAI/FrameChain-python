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

    min_input_size: Optional[Size]
    max_input_size: Optional[Size]
    _preferred_input_size: Optional[Size]
    scale_inputs: bool = False
    input_channels: int
    input_scaling_mode: ScalingMode
    standardize_input_format: bool = True

    min_output_size: Optional[Size]
    max_output_size: Optional[Size]
    _preferred_output_size: Optional[Size]
    scale_outputs: bool = False
    output_channels: int
    output_scaling_mode: ScalingMode
    standardize_output_format: bool = True

    @property
    def preferred_input_size(self) -> Optional[Size]:
        """The preferred input size for the model."""
        return self._get_preferred_size(self._preferred_input_size, self.min_input_size, self.max_input_size)

    @preferred_input_size.setter
    def preferred_input_size(self, preferred_input_size: Size):
        """
        Set the preferred input size for the model.
        Raises an error if the preferred size is outside the current bounds.
        """
        self._set_preferred_size(preferred_input_size, self.min_input_size, self.max_input_size)
        self._preferred_input_size = preferred_input_size

    @property
    def preferred_output_size(self) -> Optional[Size]:
        """The preferred output size for the model."""
        return self._get_preferred_size(self._preferred_output_size, self.min_output_size, self.max_output_size)

    @preferred_output_size.setter
    def preferred_output_size(self, preferred_output_size: Size):
        """
        Set the preferred output size for the model.
        Raises an error if the preferred size is outside the current bounds.
        """
        self._set_preferred_size(preferred_output_size, self.min_output_size, self.max_output_size)
        self._preferred_output_size = preferred_output_size

    def _get_preferred_size(self, preferred_size: Optional[Size], min_size: Optional[Size], max_size: Optional[Size]) -> Optional[Size]:
        if preferred_size is not None:
            return preferred_size
        match min_size, max_size:
            case (min_w, min_h), (max_w, max_h):
                return (min_w + max_w) // 2, (min_h + max_h) // 2
            case (min_w, min_h), _:
                return min_w, min_h
            case _, (max_w, max_h):
                return max_w, max_h
            case _, _:
                return None

    def _set_preferred_size(self, preferred_size: Size, min_size: Optional[Size], max_size: Optional[Size]):
        preferred_width, preferred_height = preferred_size
        if min_size is not None:
            min_width, min_height = min_size
            if preferred_width < min_width or preferred_height < min_height:
                raise ValueError("Preferred size is too small compared to the minimum size.")
        if max_size is not None:
            max_width, max_height = max_size
            if preferred_width > max_width or preferred_height > max_height:
                raise ValueError("Preferred size is too large compared to the maximum size.")

    def pre_run(self, **inputs: RunInput) -> RunInput:
        for k, v in inputs.items():
            if self.standardize_input_format:
                if isinstance(v, PIL.Image):
                    v = to(v, ImageFormat.np)
            if self.scale_inputs:
                v = scale(
                    v, self.min_input_size, self.max_input_size, self.input_scaling_mode
                )
            inputs[k] = v
        return inputs

    @classmethod
    def from_func(cls, func: Callable, /, **kwargs):
        return type(
            f"{func.__name__}{cls.__name__}",
            (FunctionalChain, cls),
            {"func": func, **kwargs},
        )


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

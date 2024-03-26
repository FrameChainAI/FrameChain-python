from enum import Enum
from typing import Callable, Literal, Optional, Self
from framechain.transformations.scale import ScalingMode, scale
import hashlib
from abc import ABC, abstractmethod

import stringcase
import numpy as np
from pydantic import BaseModel


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


class Chain(Runnable, Serializable, ABC):

    inputs: list[str]
    outputs: list[str]

    @classmethod
    def from_func(cls, **kwargs):
        def dec(func: Callable):
            name = kwargs.get('name', stringcase.camelcase(f"{func.__name__}{cls.__name__}"))
            bases = kwargs.get('bases', (FunctionalChain, cls))
            return type(name, bases, {"func": func, **kwargs})
        return dec

import inspect

class FunctionalChain(Chain):
    func: Callable

    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        if inspect.signature(self.func).parameters.get('self') is not None:
            return self.func(self, **inputs)
        else:
            return self.func(**inputs)

    def serialize(self) -> str:
        func_code = self.func.__code__.co_code
        return hashlib.sha256(func_code).hexdigest()


def chain(inputs: list[str], outputs: list[str], base_chain: type[Chain] = Chain, **kwargs):
    """Makes a chain from a single function.
    
    Example:
        ```python
        @chain(inputs=["image"], outputs=["image"])
        def my_chain(inputs: RunInput) -> RunOutput:
            return inputs
        ```
        
    Note: This decorator returns a `Chain` instance, not a `Chain` subclass.
            If you want a new subclass, use the `Chain.from_func` classmethod
            or consider subclassing `Chain` directly.
    """
    def dec(func: Callable):
        NewChain = base_chain.from_func(func, inputs=inputs, outputs=outputs, **kwargs)
        return NewChain()

    return dec


class ImageModel(Chain, ABC):
    model_name: str
    model_type: str

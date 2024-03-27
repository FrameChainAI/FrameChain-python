from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from pydantic import validator
from framechain.schema import BaseChain, Image, ImageType, RunInput, RunOutput, Size, convert_format
from framechain.utils.scale import ScalingMode, scale
from typingx import isinstancex

from framechain.utils.channel_format import convert_channel_format
from framechain.utils.image_type import convert_type

T = TypeVar('T')

class SimpleChain(Generic[T], BaseChain):
    """
    A chain that processes 1 input image and returns 1 output image.
    """
    
    input_name: str = "input"
    output_name: str = "output"
    
    @validator('inputs', pre=True, always=True)
    def default_inputs(cls, v):
        if not v:
            return [cls.input_name]
        return v

    @validator('outputs', pre=True, always=True)
    def default_outputs(cls, v):
        if not v:
            return [cls.output_name]
        return v
    
    def _run(self, inputs: RunInput | None, **kwargs) -> RunOutput | None:
        input = inputs[self.input_name]
        output = self._process_input(input, **kwargs)
        return {self.output_name: output}
    
    @abstractmethod
    def _process_input(self, input: T, **kwargs) -> T:
        return input

class SimpleImageChain(SimpleChain[Image]):
    pass
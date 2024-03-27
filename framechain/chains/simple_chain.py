from typing import Optional

from pydantic import validator
from framechain.schema import Chain, Image, ImageType, RunInput, RunOutput, Size, convert_format
from framechain.utils.scale import ScalingMode, scale
from typingx import isinstancex

from framechain.utils.channel_format import convert_channel_format
from framechain.utils.image_type import convert_type

class SimpleChain(Chain):
    """
    A chain that processes 1 input image and returns 1 output image.
    """
    
    input_image: str = "input"
    output_image: str = "output"
    
    @validator('inputs', pre=True, always=True)
    def default_inputs(cls, v):
        if not v:
            return [cls.input_image]
        return v

    @validator('outputs', pre=True, always=True)
    def default_outputs(cls, v):
        if not v:
            return [cls.output_image]
        return v
    
    min_input_size: Optional[Size] = None
    max_input_size: Optional[Size] = None
    preferred_input_size: Optional[Size] = None
    input_scaling_mode: Optional[ScalingMode] = None
    input_channels: Optional[int] = None
    input_type: Optional[ImageType] = None
    
    def _actual_input_size_target(self, target) -> Size:
        if target is None:
            if self.preferred_input_size is not None:
                return self.preferred_input_size
            elif self.max_input_size is not None:
                return self.max_input_size
            elif self.min_input_size is not None:
                return self.min_input_size
            else:
                raise ValueError("No target size specified")
        target = list(*target)
        if self.max_output_size is not None:
            target[0], target[1] = min(target[0], self.max_output_size[0]), min(target[1], self.max_output_size[1])
        if self.min_output_size is not None:
            target[0], target[1] = max(target[0], self.min_output_size[0]), max(target[1], self.min_output_size[1])
        return Size(*target)

    min_output_size: Optional[Size] = None
    max_output_size: Optional[Size] = None
    preferred_output_size: Optional[Size] = None
    output_scaling_mode: Optional[ScalingMode] = None
    output_channels: Optional[int] = None
    output_type: Optional[ImageType] = None
    
    def pre_run(self, inputs: RunInput) -> RunInput:
        processed_inputs = self._process_io(
            image=inputs[self.input_image],
            min_size=self.min_input_size,
            max_size=self.max_input_size,
            preferred_size=self.preferred_input_size,
            scaling_mode=self.input_scaling_mode,
            channel_format=self.input_channels,
            image_type=self.input_type
        )
        inputs.update(processed_inputs)
        return super().pre_run(inputs)

    def post_run(self, inputs: RunInput, outputs: RunOutput) -> RunOutput:
        processed_outputs = self._process_io(
            image=outputs[self.output_image],
            min_size=self.min_output_size,
            max_size=self.max_output_size,
            preferred_size=self.preferred_output_size,
            scaling_mode=self.output_scaling_mode,
            channel_format=self.standard_output_channels,
            image_type=self.standard_output_format
        )
        outputs.update(processed_outputs)
        return super().post_run(inputs, outputs)
        
    def _process_io(self, *, image: Image,
                    min_size: int, max_size: int, preferred_size: int,
                    scaling_mode: str, channel_format: int, image_type: str) -> dict:
        for k, v in image.items():
            if scaling_mode is not None:
                v = scale(
                    v,
                    min_size=min_size,
                    max_size=max_size,
                    preferred_size=preferred_size,
                    scaling_mode=scaling_mode
                )
            if channel_format is not None:
                v = convert_channel_format(v, channel_format)
            if image_type is not None:
                v = convert_type(v, image_type)
            image[k] = v
        return image


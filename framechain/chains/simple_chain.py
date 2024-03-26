from typing import Optional
from framechain.schema import Chain, Image, ImageType, RunInput, RunOutput, Size, convert_format
from framechain.transformations.scale import ScalingMode, scale
from typingx import isinstancex

from framechain.utils.channel_format import convert_channel_format
from framechain.utils.image_type import convert_type

class SimpleChain(Chain):
    
    min_input_size: Optional[Size] = None
    max_input_size: Optional[Size] = None
    preferred_input_size: Optional[Size] = None

    input_scaling_mode: Optional[ScalingMode] = None
    input_channels: Optional[int] = None
    input_type: Optional[ImageType] = None

    min_output_size: Optional[Size] = None
    max_output_size: Optional[Size] = None
    preferred_output_size: Optional[Size] = None
    
    output_scaling_mode: Optional[ScalingMode] = None
    output_channels: Optional[int] = None
    output_type: Optional[ImageType] = None
    
    def pre_run(self, inputs: RunInput) -> RunInput:
        return self._process_io(
            io=inputs,
            min_size=self.min_input_size,
            max_size=self.max_input_size,
            preferred_size=self.preferred_input_size,
            scaling_mode=self.input_scaling_mode,
            channel_format=self.input_channels,
            image_type=self.input_type
        )

    def post_run(self, inputs: RunInput, outputs: RunOutput) -> RunOutput:
        return self._process_io(
            io=outputs,
            min_size=self.min_output_size,
            max_size=self.max_output_size,
            preferred_size=self.preferred_output_size,
            scaling_mode=self.output_scaling_mode,
            channel_format=self.standard_output_channels,
            image_type=self.standard_output_format
        )
        
    def _process_io(self, *, io: dict,
                    min_size: int, max_size: int, preferred_size: int,
                    scaling_mode: str, channel_format: int, image_type: str) -> dict:
        for k, v in io.items():
            if isinstancex(v, Image):
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
                io[k] = v
        return io


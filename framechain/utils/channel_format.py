from enum import Enum
from typing import Optional



class ChannelFormat(Enum):
    L = "L"
    RGB = "RGB"
    CMYK = "CMYK"

def convert_channel_format(input, /, to: ChannelFormat, _from: Optional[ChannelFormat] = None):
    input_channel_format: ChannelFormat
    if _from is not None:
        input_channel_format = _from
    elif input.shape[2] == 1:
        input_channel_format = ChannelFormat.L
    elif input.shape[2] == 3:
        input_channel_format = ChannelFormat.RGB
    elif input.shape[2] == 4:
        input_channel_format = ChannelFormat.CMYK
    else:
        raise ValueError(f"Unsupported channel format: {output_channel_format}")
    
    output_channel_format: ChannelFormat
    
    match input_channel_format, output_channel_format:
        case ChannelFormat.L, ChannelFormat.L:
            return input
        case ChannelFormat.L, ChannelFormat.RGB:
            return input.convert("RGB")
        case ChannelFormat.L, ChannelFormat.CMYK:
            return input.convert("CMYK")
        case ChannelFormat.RGB, ChannelFormat.L:
            return input.convert("L")
        case ChannelFormat.RGB, ChannelFormat.RGB:
            return input
        case ChannelFormat.RGB, ChannelFormat.CMYK:
            return input.convert("CMYK")
        case ChannelFormat.CMYK, ChannelFormat.L:
            return input.convert("L")
        case ChannelFormat.CMYK, ChannelFormat.RGB:
            return input.convert("RGB")
        case ChannelFormat.CMYK, ChannelFormat.CMYK:
            return input


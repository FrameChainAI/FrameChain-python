from framechain.chains.decorator import chain
from framechain.schema import RunInput, RunOutput
from framechain.utils.channel_format import ChannelFormat, convert_channel_format
from framechain.utils.types import Image
from PIL import ImageOps, ImageEnhance, ImageFilter

@chain()
def flip(image: Image, horizontal: bool = True) -> Image:
    if horizontal:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)

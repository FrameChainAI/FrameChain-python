from framechain.chains.decorator import chain
from framechain.schema import RunInput, RunOutput
from framechain.utils.channel_format import ChannelFormat, convert_channel_format
from framechain.utils.types import Image
from PIL import ImageOps, ImageEnhance, ImageFilter


@chain()
def solarize(image: Image, threshold: int) -> Image:
    return ImageOps.solarize(image, threshold)

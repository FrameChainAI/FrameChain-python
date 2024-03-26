from framechain.chains.simple_chain import SimpleChain
from framechain.utils.channel_format import ChannelFormat, convert_channel_format
from framechain.utils.types import Image


@SimpleChain.from_func(standard_input_format="PIL", standard_output_format="PIL")
def greyscale(self: SimpleChain, image: Image) -> Image:
    return convert_channel_format(image, to=ChannelFormat.L)


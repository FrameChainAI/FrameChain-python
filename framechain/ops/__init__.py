from PIL import ImageOps, ImageEnhance, ImageFilter

from framechain.chains.simple_chain import SimpleChain
from framechain.schema import RunInput, RunOutput
from framechain.utils.types import Image
from framechain.utils.channel_format import ChannelFormat, convert_channel_format

class AdjustBrightness(SimpleChain):
    factor: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        enhancer = ImageEnhance.Brightness(input_image)
        output_image = enhancer.enhance(self.factor)
        return {**inputs, self.output_image: output_image}

class AdjustColor(SimpleChain):
    factor: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        enhancer = ImageEnhance.Color(input_image)
        output_image = enhancer.enhance(self.factor)
        return {**inputs, self.output_image: output_image}

class AdjustContrast(SimpleChain):
    factor: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        enhancer = ImageEnhance.Contrast(input_image)
        output_image = enhancer.enhance(self.factor)
        return {**inputs, self.output_image: output_image}

class AdjustSharpness(SimpleChain):
    factor: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        enhancer = ImageEnhance.Sharpness(input_image)
        output_image = enhancer.enhance(self.factor)
        return {**inputs, self.output_image: output_image}

class Crop(SimpleChain):
    left: int
    top: int
    right: int
    bottom: int
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.crop((self.left, self.top, self.right, self.bottom))
        return {**inputs, self.output_image: output_image}

class EdgeDetection(SimpleChain):
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.filter(ImageFilter.FIND_EDGES)
        return {**inputs, self.output_image: output_image}

class Emboss(SimpleChain):
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.filter(ImageFilter.EMBOSS)
        return {**inputs, self.output_image: output_image}

class Equalize(SimpleChain):
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = ImageOps.equalize(input_image)
        return {**inputs, self.output_image: output_image}

class Flip(SimpleChain):
    horizontal: bool = True
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        if self.horizontal:
            output_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            output_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)
        return {**inputs, self.output_image: output_image}

class GaussianBlur(SimpleChain):
    radius: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.filter(ImageFilter.GaussianBlur(self.radius))
        return {**inputs, self.output_image: output_image}


class Greyscale(SimpleChain):
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = convert_channel_format(input_image, to=ChannelFormat.L)
        return {**inputs, self.output_image: output_image}

class Posterize(SimpleChain):
    bits: int
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = ImageOps.posterize(input_image, self.bits)
        return {**inputs, self.output_image: output_image}

class Resize(SimpleChain):
    width: int
    height: int
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.resize((self.width, self.height))
        return {**inputs, self.output_image: output_image}

class Rotate(SimpleChain):
    angle: float
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.rotate(self.angle)
        return {**inputs, self.output_image: output_image}

class Solarize(SimpleChain):
    threshold: int
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = ImageOps.solarize(input_image, self.threshold)
        return {**inputs, self.output_image: output_image}

class UnsharpMask(SimpleChain):
    radius: float
    percent: int
    threshold: int
    
    def _run(self, inputs: RunInput | None) -> RunOutput | None:
        input_image = inputs[self.input_image]
        output_image = input_image.filter(ImageFilter.UnsharpMask(self.radius, self.percent, self.threshold))
        return {**inputs, self.output_image: output_image}
from enum import Enum
import PIL.Image
import numpy as np

class ImageType(Enum):
    PIL = "PIL"
    np = "np"


def convert_type(image, type: ImageType):
    if type == ImageType.PIL:
        return image if isinstance(image, PIL.Image.Image) else PIL.Image.Image.fromarray(image)
    if type == ImageType.np:
        return image if isinstance(image, np.ndarray) else np.array(image)

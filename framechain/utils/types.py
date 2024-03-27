from typing import Tuple, TypeVar

import numpy as np
import PIL.Image

T = TypeVar('T')

Size = Tuple[int, int] | Tuple[float, float]
Image = PIL.Image.Image | np.ndarray
ImageSeq = list[Image]
list2D = list[list[T]]

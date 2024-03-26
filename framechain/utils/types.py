from typing import Tuple

import numpy as np
import PIL.Image

Size = Tuple[int, int] | Tuple[float, float]
Image = PIL.Image.Image | np.ndarray
ImageSeq = list[Image]

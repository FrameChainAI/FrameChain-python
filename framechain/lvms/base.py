from abc import ABC
from framechain.schema import ImageModel


class SequentialVisionModel(ImageModel, ABC):
    output_channels: int
    context_length_limit: int

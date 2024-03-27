from abc import ABC
from framechain.schema import ImageModel


class SemanticSegmentationModel(ImageModel, ABC):
    num_classes: int
    class_names: list[str]

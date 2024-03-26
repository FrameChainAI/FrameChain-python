from abc import ABC
from framechain.schema import ImageModel


class ImageEmbeddingModel(ImageModel, ABC):
    embedding_size: int

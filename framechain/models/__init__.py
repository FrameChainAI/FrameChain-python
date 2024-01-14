from abc import ABC, abstractmethod
from typing import Literal, Self

from pydantic import BaseModel


class Serializable(ABC, BaseModel):
    type_id: str # should be unique for each model and remain constant
    version: str # should be updated when the model changes
    meta: dict # should contain any additional information about the model

    @abstractmethod
    def serialize(self) -> str:
        pass

    @abstractmethod
    @classmethod
    def deserialize(self, text: str) -> Self:
        pass


RunInput = dict
RunOutput = dict

class Runnable(ABC):
    def run(self, **inputs: RunInput) -> RunOutput|None:
        inputs = (val := self.pre_run(inputs)) if val is not None else inputs
        outputs = self._run(inputs)
        outputs = (val := self.post_run(inputs, outputs)) if val is not None else outputs
        return outputs
    
    @abstractmethod
    def pre_run(self, inputs: RunInput|None) -> RunInput|None:
        """Called before the main _run method. Good place for logging, validation, etc."""
        pass
    
    @abstractmethod
    def _run(self, inputs: RunInput|None) -> RunOutput|None:
        """The main method that does the work. Should be overridden by subclasses."""
        pass
    
    @abstractmethod
    def post_run(self, inputs: RunInput|None, outputs: RunOutput|None) -> RunOutput|None:
        """Called after the main _run method. Good place for logging, validation, etc."""
        pass
    
    def __call__(self, inputs:RunInput|None) -> RunOutput|None:
        return self.run(**inputs)
    
    def __or__(self, other):
        return SequentialRunnables(self, other)


class SequentialRunnables(Runnable):
    
    runnables: list[Runnable]
    
    def __init__(self, *runnables: list[Runnable], **kwargs):
        if 'runnables' in kwargs:
            runnables = kwargs['runnables']
        super().__init__(runnables=runnables, **kwargs)
    
    def _run(self, **kwargs) -> dict|None:
        for runnable in self.runnables:
            kwargs = runnable(**kwargs)
        return kwargs
    
    def __or__(self, other):
        self.runnables.append(other)
        return self

class Frame(Runnable, ABC):
    max_size: tuple[int, int]
    min_size: tuple[int, int]
    
    def render(self, height: int, width: int) -> np.ndarray:
        pass
    
def scale(img, max_size, min_size, mode: Literal['strict', 'crop', 'pad'] = 'strict'):
    """Scale an image to fit within a given size range."""
    
    h, w = img.shape[:2]
    min_h, min_w = min_size
    max_h, max_w = max_size
    
    h_cond = -1 if h < min_h else 1 if h > max_h else 0
    w_cond = -1 if w < min_w else 1 if w > max_w else 0
    
    match h_cond, w_cond:
        case -1, -1:
            if self.mode == 'strict':
                img = cv2.resize(img, (min_w, min_h))
            elif self.mode == 'crop':
                img = img[:min_h, :min_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, 0, min_h - h, 0, min_w - w, cv2.BORDER_CONSTANT, value=0)
        case -1, 0:
            if self.mode == 'strict':
                img = cv2.resize(img, (w, min_h))
            elif self.mode == 'crop':
                img = img[:min_h, :]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, 0, min_h - h, 0, 0, cv2.BORDER_CONSTANT, value=0)
        case -1, 1:
            if self.mode == 'strict':
                img = cv2.resize(img, (max_w, min_h))
            elif self.mode == 'crop':
                img = img[:min_h, :max_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, 0, min_h - h, 0, max_w - w, cv2.BORDER_CONSTANT, value=0)
        case 0, -1:
            if self.mode == 'strict':
                img = cv2.resize(img, (min_w, h))
            elif self.mode == 'crop':
                img = img[:, :min_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, 0, 0, 0, min_w - w, cv2.BORDER_CONSTANT, value=0)
        case 0, 0:
            pass
        case 0, 1:
            if self.mode == 'strict':
                img = cv2.resize(img, (max_w, h))
            elif self.mode == 'crop':
                img = img[:, :max_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, 0, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=0)
        case 1, -1:
            if self.mode == 'strict':
                img = cv2.resize(img, (min_w, max_h))
            elif self.mode == 'crop':
                img = img[:max_h, :min_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, max_h - h, 0, 0, min_w - w, cv2.BORDER_CONSTANT, value=0)
        case 1, 0:
            if self.mode == 'strict':
                img = cv2.resize(img, (w, max_h))
            elif self.mode == 'crop':
                img = img[:max_h, :]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, max_h - h, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)
        case 1, 1:
            if self.mode == 'strict':
                img = cv2.resize(img, (max_w, max_h))
            elif self.mode == 'crop':
                img = img[:max_h, :max_w]
            elif self.mode == 'pad':
                img = cv2.copyMakeBorder(img, max_h - h, 0, 0, max_w - w, cv2.BORDER_CONSTANT, value=0)
        case _:
            raise ValueError(f"Invalid condition: {h_cond}, {w_cond}")

    
class StaticImage(Frame, ABC):
    img: np.ndarray
    
    mode: Literal['strict', 'crop', 'pad']
    
    @property
    def max_size(self):
        return self.img.shape[:2]
    @property
    def min_size(self):
        return self.img.shape[:2]
    
    def render(self, size: tuple[int, int]) -> np.ndarray:
        return scale(self.img, size, self.mode)
    
class Grid(Frame, ABC):
    cells: list[list[Frame]]
    
    def render(self, **kwargs) -> np.ndarray:
        return np.vstack([np.hstack([cell.render(**kwargs) for cell in row]) for row in self.cells])
    
class Affine(Frame, ABC):
    frames: list[Frame]
    
    def render(self, **kwargs) -> np.ndarray:
        return np.hstack([frame.render(**kwargs) for frame in self.frames])

class Chain(Runnable, Serializable, ABC):
    pass
    
class Model(Chain, ABC):
    model_name: str
    model_type: str

class ImageModel(Model, ABC):
    min_input_size: tuple[int, int]
    max_input_size: tuple[int, int]
    input_channels: int

class ImageEmbeddingModel(ImageModel, ABC):
    embedding_size: int

class PoseEstimationModel(ImageModel, ABC):
    pass

class InstanceSegmentationModel(ImageModel, ABC):
    pass

class SemanticSegmentationModel(ImageModel, ABC):
    num_classes: int
    class_names: list[str]

class LargeVisionModel(ImageModel, ABC):
    output_channels: int
    context_length_limit: int
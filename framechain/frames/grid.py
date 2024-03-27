from typing import Optional

from pydantic import validator
from framechain.schema import BaseChain, Image, RunInput, RunOutput
from framechain.utils.types import list2D

class GridMerge(BaseChain):
    input_names: list2D[str]
    output_name: str
    
    vert_split_weights: Optional[list[float]] = None
    horz_split_weights: Optional[list[float]] = None
    
    @validator('input_names')
    def _validate_input_names_structure(cls, v):
        if not all(isinstance(row, list) for row in v):
            raise ValueError("input_names must be a list of lists")
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("input_names must be a non-jagged 2D list")
        return v

    @validator('vert_split_weights', 'horz_split_weights', always=True)
    def _validate_split_weights_lengths(cls, v, values, **kwargs):
        if 'input_names' not in values:
            return v  # Can't validate without input_names
        input_names = values['input_names']
        if kwargs['field'].name == 'vert_split_weights' and v is not None and len(v) != len(input_names):
            raise ValueError("Length of vert_split_weights must match the number of rows in input_names")
        if kwargs['field'].name == 'horz_split_weights' and v is not None and len(v) != len(input_names[0]):
            raise ValueError("Length of horz_split_weights must match the number of columns in input_names")
        return v
    
    def _run(self, inputs: RunInput) -> RunOutput:
        input_images = [inputs[name] for name in self.input_names]
        from PIL import ImageOps

        # Determine the total weight for vertical and horizontal splits
        total_vert_weight = sum(self.vert_split_weights) if self.vert_split_weights else len(input_images)
        total_horz_weight = sum(self.horz_split_weights) if self.horz_split_weights else len(input_images[0])

        # Calculate the size of each image segment based on weights
        vert_size, horz_size = self.
        vert_sizes = [int(self.input_size[1] * (weight / total_vert_weight)) for weight in self.vert_split_weights] if self.vert_split_weights else [self.input_size[1] // len(input_images) for _ in input_images]
        horz_sizes = [int(self.input_size[0] * (weight / total_horz_weight)) for weight in self.horz_split_weights] if self.horz_split_weights else [self.input_size[0] // len(input_images[0]) for _ in input_images[0]]

        # Resize and merge images
        merged_image = Image.new('RGB', self.input_size)
        y_offset = 0
        for row, vert_size in zip(input_images, vert_sizes):
            x_offset = 0
            for img_name, horz_size in zip(row, horz_sizes):
                img = inputs[img_name]
                # Resize image based on calculated segment size
                img_resized = img.resize((horz_size, vert_size), Image.ANTIALIAS)
                # Paste resized image into the correct position
                merged_image.paste(img_resized, (x_offset, y_offset))
                x_offset += horz_size
            y_offset += vert_size

        return {self.output_name: merged_image}


class GridSplit(Split):
    input_name: str
    output_names: list2D[str]
    

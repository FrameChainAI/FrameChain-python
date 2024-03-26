from typing import Optional

from pydantic import validator
from framechain.chains.merge import Merge
from framechain.chains.split import Split
from framechain.schema import Chain, Image

# type Struct1D[T_struct, T_item] = T_struct[T_item]
# type Struct2D[T_struct, T_item] = T_struct[T_struct[T_item]]
# type Grid[T] = Struct2D[list|tuple, T]

class GridMerge(Merge):
    input_names: list[list[str]]
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
    
    @property
    def _vert_split_weights(self) -> list[float]:
        return self.vert_split_weights or len(self.input_names)*[1.0]
    @property
    def _horz_split_weights(self) -> list[float]:
        return self.horz_split_weights or len(self.input_names[0])*[1.0]

    def _run(self, inputs: dict[str, Image]) -> Image:
        # Validate input_names structure
        if not all(len(row) == len(self.input_names[0]) for row in self.input_names):
            raise ValueError("input_names must be a non-jagged 2D list")

        # Validate vert_split_weights and horz_split_weights lengths
        if self.vert_split_weights and len(self.vert_split_weights) != len(self.input_names):
            raise ValueError("Length of vert_split_weights must match the number of rows in input_names")
        if self.horz_split_weights and len(self.horz_split_weights) != len(self.input_names[0]):
            raise ValueError("Length of horz_split_weights must match the number of columns in input_names")

        # Merge images vertically within each row
        row_images = []


class GridSplit(Split):
    input_name: str
    output_names: Grid[str]
    

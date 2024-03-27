
from typing import Callable

from framechain.chains.simple_chain import SimpleChain
from framechain.schema import Chain


def chain(dict_mode=False, inputs: list[str]=None, outputs: list[str]=None, base_chain: type[Chain] = SimpleChain, **kwargs):
    """Makes a chain from a single function.
    
    Example:
        ```python
        @chain(inputs=["image"], outputs=["image"])
        def my_chain(inputs: RunInput) -> RunOutput:
            return inputs
        ```
        
    Note: This decorator returns a `Chain` instance, not a `Chain` subclass.
            If you want a new subclass, use the `Chain.from_func` classmethod
            or consider subclassing `Chain` directly.
    """
    def dec(func: Callable):
        dec = SimpleChain.from_func(dict_mode=dict_mode, inputs=inputs, outputs=outputs, **kwargs)
        cls = dec(func)
        return cls(**kwargs)

    return dec
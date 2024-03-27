
from typing import Callable

from framechain.chains.simple_chain import SimpleChain
from framechain.schema import Chain


def chain(inputs: list[str], outputs: list[str], base_chain: type[Chain] = SimpleChain, **kwargs):
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
        NewChain = SimpleChain.from_func(func, inputs=inputs, outputs=outputs, **kwargs)
        return NewChain()

    return dec
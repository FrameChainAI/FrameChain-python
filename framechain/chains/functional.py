import hashlib
import inspect
import json
from typing import Callable, Self

from pydantic import BaseModel

from framechain.schema import Chain, RunInput, RunOutput

class FunctionalChain(Chain):
    _func: Callable[[RunInput], RunOutput]
    _func_hash: str

    def _run(self, inputs: RunInput) -> RunOutput:
        if inspect.signature(self._func).parameters.get('self') is not None:
            return self._func(self, **inputs)
        else:
            return self._func(**inputs)

    def serialize(self) -> str:
        # sign the func
        func_code = self._func.__code__.co_code
        self._func_hash = hashlib.sha256(func_code).hexdigest()
        # actually serialize
        serializable_model = self.model_dump_json()
        return json.dumps(serializable_model)

    @classmethod
    def deserialize(cls, text: str) -> Self:
        instance = cls.model_validate_json(text)

        # verify the func
        actual_func_hash = hashlib.sha256(instance._func.__code__.co_code).hexdigest()
        if instance._func_hash != actual_func_hash:
            raise ValueError("Saved function hash does not match the code! It may have been updated since last used or maliciously modified.")
        
        return instance
